// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \brief Contains the classes required to extend the black-oil model by MICP.
 */
#ifndef OPM_BLACK_OIL_MICP_MODULE_HH
#define OPM_BLACK_OIL_MICP_MODULE_HH

#include <dune/common/fvector.hh>

#include <opm/models/blackoil/blackoilmicpparams.hpp>
#include <opm/models/blackoil/blackoilproperties.hh>

#include <opm/models/io/vtkblackoilmicpmodule.hpp>

#include <cstddef>
#include <stdexcept>

namespace Opm {
/*!
 * \ingroup BlackOil
 * \brief Contains the high level supplements required to extend the black oil
 *        model by MICP.
 */
template <class TypeTag, bool enableMICPV = getPropValue<TypeTag, Properties::EnableMICP>()>
class BlackOilMICPModule
{
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using PrimaryVariables = GetPropType<TypeTag, Properties::PrimaryVariables>;
    using IntensiveQuantities = GetPropType<TypeTag, Properties::IntensiveQuantities>;
    using ExtensiveQuantities = GetPropType<TypeTag, Properties::ExtensiveQuantities>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using Model = GetPropType<TypeTag, Properties::Model>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using EqVector = GetPropType<TypeTag, Properties::EqVector>;
    using RateVector = GetPropType<TypeTag, Properties::RateVector>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;
    using Problem = GetPropType<TypeTag, Properties::Problem>;

    using Toolbox = MathToolbox<Evaluation>;

    using TabulatedFunction = typename BlackOilMICPParams<Scalar>::TabulatedFunction;

    enum { waterCompIdx = FluidSystem::waterCompIdx };

    static constexpr unsigned microbialConcentrationIdx = Indices::microbialConcentrationIdx;
    static constexpr unsigned oxygenConcentrationIdx = Indices::oxygenConcentrationIdx;
    static constexpr unsigned ureaConcentrationIdx = Indices::ureaConcentrationIdx;
    static constexpr unsigned biofilmConcentrationIdx = Indices::biofilmConcentrationIdx;
    static constexpr unsigned calciteConcentrationIdx = Indices::calciteConcentrationIdx;
    static constexpr unsigned contiMicrobialEqIdx = Indices::contiMicrobialEqIdx;
    static constexpr unsigned contiOxygenEqIdx = Indices::contiOxygenEqIdx;
    static constexpr unsigned contiUreaEqIdx = Indices::contiUreaEqIdx;
    static constexpr unsigned contiBiofilmEqIdx = Indices::contiBiofilmEqIdx;
    static constexpr unsigned contiCalciteEqIdx = Indices::contiCalciteEqIdx;
    static constexpr unsigned waterPhaseIdx = FluidSystem::waterPhaseIdx;

    static constexpr unsigned enableMICP = enableMICPV;

    static constexpr unsigned numEq = getPropValue<TypeTag, Properties::NumEq>();

public:
    //! \brief Set parameters.
    static void setParams(BlackOilMICPParams<Scalar>&& params)
    {
        params_ = params;
    }

    /*!
     * \brief Register all run-time parameters for the black-oil MICP module.
     */
    static void registerParameters()
    {
        if constexpr (enableMICP)
            VtkBlackOilMICPModule<TypeTag>::registerParameters();
    }

    /*!
     * \brief Register all MICP specific VTK and ECL output modules.
     */
    static void registerOutputModules(Model& model,
                                      Simulator& simulator)
    {
        if constexpr (enableMICP)
            model.addOutputModule(new VtkBlackOilMICPModule<TypeTag>(simulator));
    }

    static bool eqApplies(unsigned eqIdx)
    {
        if constexpr (enableMICP)
            return eqIdx == contiMicrobialEqIdx || eqIdx == contiOxygenEqIdx || eqIdx == contiUreaEqIdx 
                   || eqIdx == contiBiofilmEqIdx || eqIdx == contiCalciteEqIdx;
        else
            return false;
    }

    static Scalar eqWeight([[maybe_unused]] unsigned eqIdx)
    {
        assert(eqApplies(eqIdx));

        // TODO: it may be beneficial to chose this differently.
        return static_cast<Scalar>(1.0);
    }

    // must be called after water storage is computed
    template <class LhsEval>
    static void addStorage(Dune::FieldVector<LhsEval, numEq>& storage,
                           const IntensiveQuantities& intQuants)
    {
        if constexpr (enableMICP) {
            const auto& fs = intQuants.fluidState();
            LhsEval surfaceVolumeWater = Toolbox::template decay<LhsEval>(fs.invB(waterPhaseIdx))
                                         * Toolbox::template decay<LhsEval>(intQuants.porosity());
            // avoid singular matrix if no water is present.
            surfaceVolumeWater = max(surfaceVolumeWater, 1e-10);
            // suspended microbes in water phase
            const LhsEval massMicrobes = surfaceVolumeWater * Toolbox::template decay<LhsEval>(intQuants.microbialConcentration());
            LhsEval accumulationMicrobes = massMicrobes;
            storage[contiMicrobialEqIdx] += accumulationMicrobes;
            // oxygen in water phase
            const LhsEval massOxygen = surfaceVolumeWater * Toolbox::template decay<LhsEval>(intQuants.oxygenConcentration());
            LhsEval accumulationOxygen = massOxygen;
            storage[contiOxygenEqIdx] += accumulationOxygen;
            // urea in water phase
            const LhsEval massUrea = surfaceVolumeWater * Toolbox::template decay<LhsEval>(intQuants.ureaConcentration());
            LhsEval accumulationUrea = massUrea;
            storage[contiUreaEqIdx] += accumulationUrea;
            // biofilm
            const LhsEval massBiofilm = Toolbox::template decay<LhsEval>(intQuants.biofilmConcentration());
            LhsEval accumulationBiofilm = massBiofilm;
            storage[contiBiofilmEqIdx] += accumulationBiofilm;
            // calcite
            const LhsEval massCalcite = Toolbox::template decay<LhsEval>(intQuants.calciteConcentration());
            LhsEval accumulationCalcite = massCalcite;
            storage[contiCalciteEqIdx] += accumulationCalcite;
        }
    }

    template <class UpEval, class Eval, class IntensiveQuantities>
    static void addMICPFluxes_(RateVector& flux,
                               const Eval& volumeFlux,
                               const IntensiveQuantities& upFs)
    {
        if constexpr (enableMICP) {
            flux[contiMicrobialEqIdx] +=
                decay<UpEval>(upFs.microbialConcentration())
                * volumeFlux;
            flux[contiOxygenEqIdx] +=
                decay<UpEval>(upFs.oxygenConcentration())
                * volumeFlux;
            flux[contiUreaEqIdx] +=
                decay<UpEval>(upFs.ureaConcentration())
                * volumeFlux;
        }
    }

    static void computeFlux([[maybe_unused]] RateVector& flux,
                            [[maybe_unused]] const ElementContext& elemCtx,
                            [[maybe_unused]] unsigned scvfIdx,
                            [[maybe_unused]] unsigned timeIdx)

    {
        if constexpr (enableMICP) {
            flux[contiMicrobialEqIdx] = 0.0;
            flux[contiOxygenEqIdx] = 0.0;
            flux[contiUreaEqIdx] = 0.0;
            const auto& extQuants = elemCtx.extensiveQuantities(scvfIdx, timeIdx);
            unsigned focusIdx = elemCtx.focusDofIndex();
            unsigned upIdx = extQuants.upstreamIndex(waterPhaseIdx);
            if (upIdx == focusIdx)
                addMICPFluxes_<Evaluation>(flux, elemCtx, scvfIdx, timeIdx);
            else
                addMICPFluxes_<Scalar>(flux, elemCtx, scvfIdx, timeIdx);
        }
    }

    template <class UpstreamEval>
    static void addMICPFluxes_(RateVector& flux,
                               const ElementContext& elemCtx,
                               unsigned scvfIdx,
                               unsigned timeIdx)
    {
        const auto& extQuants = elemCtx.extensiveQuantities(scvfIdx, timeIdx);
        unsigned upIdx = extQuants.upstreamIndex(waterPhaseIdx);
        const auto& up = elemCtx.intensiveQuantities(upIdx, timeIdx);
        const auto& fs = up.fluidState();
        const auto& volFlux = extQuants.volumeFlux(waterPhaseIdx);
        addMICPFluxes_<UpstreamEval>(flux, volFlux, fs);
    }

    // see https://doi.org/10.1016/j.ijggc.2021.103256 for the MICP processes in the model
    static void addSource(RateVector& source,
                          const Problem& problem,
                          const IntensiveQuantities& intQuants,
                          unsigned globalSpaceIdex)
    {
        if constexpr (enableMICP) {
            const auto& velocityInf = problem.model().linearizer().getVelocityInfo();
            auto velocityInfos = velocityInf[globalSpaceIdex];
            Scalar normVelocityCell = 0.0;
            for (const auto& velocityInfo : velocityInfos) {
                normVelocityCell = max(normVelocityCell, std::abs(velocityInfo.velocity[waterPhaseIdx]));
            }

            // get the model parameters
            const auto b = intQuants.fluidState().invB(waterPhaseIdx);
            unsigned satnumIdx = problem.satnumRegionIndex(globalSpaceIdex);
            Scalar k_a = microbialAttachmentRate(satnumIdx);
            Scalar k_d = microbialDeathRate(satnumIdx);
            Scalar rho_b = densityBiofilm(satnumIdx);
            Scalar rho_c = densityCalcite(satnumIdx);
            Scalar k_str = detachmentRate(satnumIdx);
            Scalar eta = detachmentExponent(satnumIdx);
            Scalar k_o = halfVelocityOxygen(satnumIdx);
            Scalar k_u = halfVelocityUrea(satnumIdx) / 10;// dividing by scaling factor 10 (see WellInterfaceGeneric.cpp)
            Scalar mu = maximumGrowthRate(satnumIdx);
            Scalar mu_u = maximumUreaUtilization(satnumIdx) / 10;// dividing by scaling factor 10
            Scalar Y_sb = yieldGrowthCoefficient(satnumIdx);
            Scalar F = oxygenConsumptionFactor(satnumIdx);
            Scalar Y_uc = yieldUreaToCalciteCoefficient(satnumIdx) * 10;// multiplying by scaling factor 10

            // compute the processes
            source[Indices::contiMicrobialEqIdx] += intQuants.microbialConcentration() * intQuants.porosity() * b *
                                                        (Y_sb * mu * intQuants.oxygenConcentration() / (k_o + intQuants.oxygenConcentration()) - k_d - k_a)
                                                        + rho_b * intQuants.biofilmConcentration() * k_str * pow(normVelocityCell, eta);

            source[Indices::contiOxygenEqIdx] -= (intQuants.microbialConcentration() * intQuants.porosity() * b + rho_b * intQuants.biofilmConcentration()) *
                                                        F * mu * intQuants.oxygenConcentration() / (k_o + intQuants.oxygenConcentration());

            source[Indices::contiUreaEqIdx] -= rho_b * intQuants.biofilmConcentration() * mu_u * intQuants.ureaConcentration() / (k_u + intQuants.ureaConcentration());

            source[Indices::contiBiofilmEqIdx] += intQuants.biofilmConcentration() * (Y_sb * mu * intQuants.oxygenConcentration() / (k_o + intQuants.oxygenConcentration()) - k_d
                                                            - k_str * pow(normVelocityCell, eta) - Y_uc * (rho_b / rho_c) * intQuants.biofilmConcentration() * mu_u *
                                                                    (intQuants.ureaConcentration() / (k_u + intQuants.ureaConcentration())) / (intQuants.porosity() + intQuants.biofilmConcentration()))
                                                + k_a * intQuants.microbialConcentration() * intQuants.porosity() * b / rho_b;

            source[Indices::contiCalciteEqIdx] += (rho_b / rho_c) * intQuants.biofilmConcentration() * Y_uc * mu_u * intQuants.ureaConcentration() / (k_u + intQuants.ureaConcentration());
        }
    }

    static void addSource([[maybe_unused]] RateVector& source,
                          [[maybe_unused]] const ElementContext& elemCtx,
                          [[maybe_unused]] unsigned dofIdx,
                          [[maybe_unused]] unsigned timeIdx)
    {
        if constexpr (enableMICP) {
            const auto& problem = elemCtx.problem();
            const auto& intQuants = elemCtx.intensiveQuantities(dofIdx, timeIdx);
            addSource(source, problem, intQuants, dofIdx);
        }
    }

    static const Scalar densityBiofilm(unsigned satnumRegionIdx)
    {
        return params_.densityBiofilm_[satnumRegionIdx];
    }

    static const Scalar densityCalcite(unsigned satnumRegionIdx)
    {
        return params_.densityCalcite_[satnumRegionIdx];
    }

    static const Scalar detachmentRate(unsigned satnumRegionIdx)
    {
        return params_.detachmentRate_[satnumRegionIdx];
    }

    static const Scalar detachmentExponent(unsigned satnumRegionIdx)
    {
        return params_.detachmentExponent_[satnumRegionIdx];
    }

    static const Scalar halfVelocityOxygen(unsigned satnumRegionIdx)
    {
        return params_.halfVelocityOxygen_[satnumRegionIdx];
    }

    static const Scalar halfVelocityUrea(unsigned satnumRegionIdx)
    {
        return params_.halfVelocityUrea_[satnumRegionIdx];
    }

    static const Scalar maximumGrowthRate(unsigned satnumRegionIdx)
    {
        return params_.maximumGrowthRate_[satnumRegionIdx];
    }

    static const Scalar maximumUreaUtilization(unsigned satnumRegionIdx)
    {
        return params_.maximumUreaUtilization_[satnumRegionIdx];
    }

    static const Scalar microbialAttachmentRate(unsigned satnumRegionIdx)
    {
        return params_.microbialAttachmentRate_[satnumRegionIdx];
    }

    static const Scalar microbialDeathRate(unsigned satnumRegionIdx)
    {
        return params_.microbialDeathRate_[satnumRegionIdx];
    }

    static const Scalar oxygenConsumptionFactor(unsigned satnumRegionIdx)
    {
        return params_.oxygenConsumptionFactor_[satnumRegionIdx];
    }

    static const Scalar yieldGrowthCoefficient(unsigned satnumRegionIdx)
    {
        return params_.yieldGrowthCoefficient_[satnumRegionIdx];
    }

    static const Scalar yieldUreaToCalciteCoefficient(unsigned satnumRegionIdx)
    {
        return params_.yieldUreaToCalciteCoefficient_[satnumRegionIdx];
    }

    static const Scalar microbialDiffusion(unsigned pvtRegionIdx)
    {
        return params_.microbialDiffusion_[pvtRegionIdx];
    }

    static const Scalar oxygenDiffusion(unsigned pvtRegionIdx)
    {
        return params_.oxygenDiffusion_[pvtRegionIdx];
    }

    static const Scalar ureaDiffusion(unsigned pvtRegionIdx)
    {
        return params_.ureaDiffusion_[pvtRegionIdx];
    }

    static const TabulatedFunction& permfactTable(const ElementContext& elemCtx,
                                                  unsigned scvIdx,
                                                  unsigned timeIdx)
    {
        unsigned satnumRegionIdx = elemCtx.problem().satnumRegionIndex(elemCtx, scvIdx, timeIdx);
        return params_.permfactTable_[satnumRegionIdx];
    }

    static const TabulatedFunction& permfactTable(unsigned satnumRegionIdx)
    {
        return params_.permfactTable_[satnumRegionIdx];
    }

private:
    static BlackOilMICPParams<Scalar> params_;
};


template <class TypeTag, bool enableMICPV>
BlackOilMICPParams<typename BlackOilMICPModule<TypeTag, enableMICPV>::Scalar>
BlackOilMICPModule<TypeTag, enableMICPV>::params_;

/*!
 * \ingroup BlackOil
 * \class Opm::BlackOilMICPIntensiveQuantities
 *
 * \brief Provides the volumetric quantities required for the equations needed by the
 *        MICP extension of the black-oil model.
 */
template <class TypeTag, bool enableMICPV = getPropValue<TypeTag, Properties::EnableMICP>()>
class BlackOilMICPIntensiveQuantities
{
    using Implementation = GetPropType<TypeTag, Properties::IntensiveQuantities>;

    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using PrimaryVariables = GetPropType<TypeTag, Properties::PrimaryVariables>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;

    using MICPModule = BlackOilMICPModule<TypeTag>;

    static constexpr int microbialConcentrationIdx = Indices::microbialConcentrationIdx;
    static constexpr int oxygenConcentrationIdx = Indices::oxygenConcentrationIdx;
    static constexpr int ureaConcentrationIdx = Indices::ureaConcentrationIdx;
    static constexpr int biofilmConcentrationIdx = Indices::biofilmConcentrationIdx;
    static constexpr int calciteConcentrationIdx = Indices::calciteConcentrationIdx;
    static constexpr int waterPhaseIdx = FluidSystem::waterPhaseIdx;


public:

    /*!
     * \brief Update the intensive properties needed to handle MICP from the
     *        primary variables
     *
     */
    void MICPPropertiesUpdate_(const ElementContext& elemCtx,
                               unsigned dofIdx,
                               unsigned timeIdx)
    {
        const auto linearizationType = elemCtx.linearizationType();
        const PrimaryVariables& priVars = elemCtx.primaryVars(dofIdx, timeIdx);
        const Scalar referencePorosity_ = elemCtx.problem().referencePorosity(dofIdx, timeIdx);

        microbialConcentration_ = priVars.makeEvaluation(microbialConcentrationIdx, timeIdx, linearizationType);
        oxygenConcentration_ = priVars.makeEvaluation(oxygenConcentrationIdx, timeIdx, linearizationType);
        ureaConcentration_ = priVars.makeEvaluation(ureaConcentrationIdx, timeIdx, linearizationType);
        biofilmConcentration_ = priVars.makeEvaluation(biofilmConcentrationIdx, timeIdx, linearizationType);
        calciteConcentration_ = priVars.makeEvaluation(calciteConcentrationIdx, timeIdx, linearizationType);

        const Evaluation porosityFactor  = min(1.0 - (biofilmConcentration_+calciteConcentration_)/(referencePorosity_+1e-8), 1.0); //phi/phi_0
        unsigned satnumRegionIdx = elemCtx.problem().satnumRegionIndex(elemCtx, dofIdx, timeIdx);
        const auto& permfactTable = MICPModule::permfactTable(satnumRegionIdx);
        permFactor_ = permfactTable.eval(porosityFactor, /*extrapolation=*/true);

        biofilmMass_ = referencePorosity_*biofilmConcentration_*MICPModule::densityBiofilm(satnumRegionIdx);
        calciteMass_ = referencePorosity_*calciteConcentration_*MICPModule::densityCalcite(satnumRegionIdx);
    }

    const Evaluation& microbialConcentration() const
    { return microbialConcentration_; }

    const Evaluation& oxygenConcentration() const
    { return oxygenConcentration_; }

    const Evaluation& ureaConcentration() const
    { return ureaConcentration_; }

    const Evaluation& biofilmConcentration() const
    { return biofilmConcentration_; }

    const Evaluation& calciteConcentration() const
    { return calciteConcentration_; }

    const Evaluation biofilmMass() const
    { return biofilmMass_; }

    const Evaluation calciteMass() const
    { return calciteMass_; }

    const Evaluation& permFactor() const
    { return permFactor_; }

protected:
    Evaluation microbialConcentration_;
    Evaluation oxygenConcentration_;
    Evaluation ureaConcentration_;
    Evaluation biofilmConcentration_;
    Evaluation calciteConcentration_;
    Evaluation biofilmMass_;
    Evaluation calciteMass_;
    Evaluation permFactor_;

};

template <class TypeTag>
class BlackOilMICPIntensiveQuantities<TypeTag, false>
{
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;

public:
    void MICPPropertiesUpdate_(const ElementContext&,
                               unsigned,
                               unsigned)
    { }

    const Evaluation& microbialConcentration() const
    { throw std::logic_error("microbialConcentration() called but MICP is disabled"); }

    const Evaluation& oxygenConcentration() const
    { throw std::logic_error("oxygenConcentration() called but MICP is disabled"); }

    const Evaluation& ureaConcentration() const
    { throw std::logic_error("ureaConcentration() called but MICP is disabled"); }

    const Evaluation& biofilmConcentration() const
    { throw std::logic_error("biofilmConcentration() called but MICP is disabled"); }

    const Evaluation& calciteConcentration() const
    { throw std::logic_error("calciteConcentration() called but MICP is disabled"); }

    const Evaluation& biofilmMass() const
    { throw std::logic_error("biofilmMass() called but MICP is disabled"); }

    const Evaluation& calciteMass() const
    { throw std::logic_error("calciteMass() called but MICP is disabled"); }

    const Evaluation& permFactor() const
    { throw std::logic_error("permFactor() called but MICP is disabled"); }
};

/*!
 * \ingroup BlackOil
 * \class Opm::BlackOilMICPExtensiveQuantities
 *
 * \brief Provides the MICP specific extensive quantities to the generic black-oil
 *        module's extensive quantities.
 */
template <class TypeTag, bool enableMICPV = getPropValue<TypeTag, Properties::EnableMICP>()>
class BlackOilMICPExtensiveQuantities
{
    using Implementation = GetPropType<TypeTag, Properties::ExtensiveQuantities>;
};

template <class TypeTag>
class BlackOilMICPExtensiveQuantities<TypeTag, false>{};

} // namespace Opm

#endif
