/*
  Copyright 2016 SINTEF ICT, Applied Mathematics.
  Copyright 2016 - 2017 Statoil ASA.
  Copyright 2017 Dr. Blatt - HPC-Simulation-Software & Services
  Copyright 2016 - 2018 IRIS AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <config.h>
#include <opm/simulators/wells/BlackoilWellModelRestart.hpp>

#include <opm/output/data/Groups.hpp>

#include <opm/simulators/wells/BlackoilWellModelGeneric.hpp>
#include <opm/simulators/wells/PerforationData.hpp>
#include <opm/simulators/wells/SingleWellState.hpp>

namespace Opm {

void BlackoilWellModelRestart::
loadRestartConnectionData(const std::vector<data::Rates::opt>& phs,
                          const data::Well&                    rst_well,
                          const std::vector<PerforationData>&  old_perf_data,
                          SingleWellState&                     ws) const
{
    auto& perf_data        = ws.perf_data;
    auto  perf_pressure    = perf_data.pressure.begin();
    auto  perf_rates       = perf_data.rates.begin();
    auto  perf_phase_rates = perf_data.phase_rates.begin();

    for (const auto& pd : old_perf_data) {
        const auto& rst_connection = rst_well.connections[pd.ecl_index];

        *perf_pressure = rst_connection.pressure;       ++perf_pressure;
        *perf_rates    = rst_connection.reservoir_rate; ++perf_rates;

        for (const auto& phase : phs) {
            *perf_phase_rates = rst_connection.rates.get(phase);
            ++perf_phase_rates;
        }
    }
}

void BlackoilWellModelRestart::
loadRestartSegmentData(const std::string&                   well_name,
                       const std::vector<data::Rates::opt>& phs,
                       const data::Well&                    rst_well,
                       SingleWellState&                     ws) const
{
    const auto& segment_set = wellModel_.getWellEcl(well_name).getSegments();
    const auto& rst_segments = rst_well.segments;

    // \Note: Eventually we need to handle the situations that some segments are shut
    assert(0u + segment_set.size() == rst_segments.size());

    const auto np = phs.size();
    const auto pres_idx = data::SegmentPressures::Value::Pressure;

    auto& segments = ws.segments;
    auto& segment_pressure = segments.pressure;
    auto& segment_rates = segments.rates;
    for (const auto& [segNum, rst_segment] : rst_segments) {
        const int segment_index = segment_set.segmentNumberToIndex(segNum);

        // Recovering segment rates and pressure from the restart values
        segment_pressure[segment_index] = rst_segment.pressures[pres_idx];

        const auto& rst_segment_rates = rst_segment.rates;
        for (auto p = 0*np; p < np; ++p) {
            segment_rates[segment_index*np + p] = rst_segment_rates.get(phs[p]);
        }
    }
}

void BlackoilWellModelRestart::
loadRestartWellData(const std::string&                   well_name,
                    const bool                           handle_ms_well,
                    const std::vector<data::Rates::opt>& phs,
                    const data::Well&                    rst_well,
                    const std::vector<PerforationData>&  old_perf_data,
                    SingleWellState&                     ws) const
{
    const auto np = phs.size();

    ws.bhp = rst_well.bhp;
    ws.thp = rst_well.thp;
    ws.temperature = rst_well.temperature;

    if (rst_well.current_control.isProducer) {
        ws.production_cmode = rst_well.current_control.prod;
    }
    else {
        ws.injection_cmode = rst_well.current_control.inj;
    }

    for (auto i = 0*np; i < np; ++i) {
        assert( rst_well.rates.has( phs[ i ] ) );
        ws.surface_rates[i] = rst_well.rates.get(phs[i]);
    }

    this->loadRestartConnectionData(phs, rst_well, old_perf_data, ws);

    if (handle_ms_well && !rst_well.segments.empty()) {
        this->loadRestartSegmentData(well_name, phs, rst_well, ws);
    }
}

void BlackoilWellModelRestart::
loadRestartGroupData(const std::string&     group,
                     const data::GroupData& value,
                     GroupState& grpState) const
{
    using GPMode = Group::ProductionCMode;
    using GIMode = Group::InjectionCMode;

    const auto cpc = value.currentControl.currentProdConstraint;
    const auto cgi = value.currentControl.currentGasInjectionConstraint;
    const auto cwi = value.currentControl.currentWaterInjectionConstraint;

    if (cpc != GPMode::NONE) {
        grpState.production_control(group, cpc);
    }

    if (cgi != GIMode::NONE) {
        grpState.injection_control(group, Phase::GAS, cgi);
    }

    if (cwi != GIMode::NONE) {
        grpState.injection_control(group, Phase::WATER, cwi);
    }
}



} // namespace Opm
