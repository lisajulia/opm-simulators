/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.

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



#include <opm/core/pressure/FlowBCManager.hpp>

#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/wells.h>
#include <opm/core/wells/WellsManager.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/core/simulator/initState.hpp>
#include <opm/core/simulator/SimulatorReport.hpp>
#include <opm/core/simulator/SimulatorTimer.hpp>
#include <opm/core/utility/miscUtilities.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>

#include <opm/core/io/eclipse/EclipseWriter.hpp>
#include <opm/core/props/IncompPropertiesBasic.hpp>
#include <opm/core/props/IncompPropertiesFromDeck.hpp>
#include <opm/core/props/rock/RockCompressibility.hpp>

#include <opm/core/linalg/LinearSolverFactory.hpp>

#include <opm/core/simulator/WellState.hpp>

#include <opm/polymer/fullyimplicit/SimulatorFullyImplicitTwophasePolymer.hpp>
#include <opm/polymer/fullyimplicit/IncompPropsAdInterface.hpp>
#include <opm/polymer/fullyimplicit/IncompPropsAdBasic.hpp>
#include <opm/polymer/fullyimplicit/IncompPropsAdFromDeck.hpp>
#include <opm/polymer/fullyimplicit/PolymerPropsAd.hpp>
#include <opm/polymer/PolymerProperties.hpp>
#include <opm/polymer/PolymerInflow.hpp>
#include <opm/polymer/PolymerState.hpp>
#include <opm/core/utility/share_obj.hpp>

#include <boost/scoped_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>


namespace
{
    void warnIfUnusedParams(const Opm::parameter::ParameterGroup& param)
    {
        if (param.anyUnused()) {
            std::cout << "--------------------   Unused parameters:   --------------------\n";
            param.displayUsage();
            std::cout << "----------------------------------------------------------------" << std::endl;
        }
    }
} // anon namespace



// ----------------- Main program -----------------
int
main(int argc, char** argv)
try
{
    using namespace Opm;

    std::cout << "\n================    Test program for fully implicit three-phase black-oil flow     ===============\n\n";
    parameter::ParameterGroup param(argc, argv, false);
    std::cout << "---------------    Reading parameters     ---------------" << std::endl;

    // If we have a "deck_filename", grid and props will be read from that.
    bool use_deck = param.has("deck_filename");
    if (!use_deck) {
        OPM_THROW(std::runtime_error, "This program must be run with an input deck. "
                  "Specify the deck with deck_filename=deckname.data (for example).");
    }
    boost::scoped_ptr<EclipseGridParser> deck;
    boost::scoped_ptr<GridManager> grid;
    boost::scoped_ptr<IncompPropertiesInterface> props;
    boost::scoped_ptr<IncompPropsAdInterface> new_props;
//    boost::scoped_ptr<PolymerPropsAd> polymer_props;
    PolymerState state;
    // bool check_well_controls = false;
    // int max_well_control_iterations = 0;
    double gravity[3] = { 0.0 };
    std::string deck_filename = param.get<std::string>("deck_filename");
    deck.reset(new EclipseGridParser(deck_filename));
    // Grid init
    grid.reset(new GridManager(*deck));

    // use the capitalized part of the deck's filename between the
    // last '/' and the last '.' character as base name.
    std::string baseName = deck_filename;
    auto charPos = baseName.rfind('/');
    if (charPos != std::string::npos)
        baseName = baseName.substr(charPos + 1);
    charPos = baseName.rfind('.');
    if (charPos != std::string::npos)
        baseName = baseName.substr(0, charPos);
    baseName = boost::to_upper_copy(baseName);

    // Rock and fluid init
    props.reset(new IncompPropertiesFromDeck(*deck, *grid->c_grid()));
    new_props.reset(new IncompPropsAdFromDeck(*deck, *grid->c_grid()));
    PolymerProperties polymer_props(*deck);
    PolymerPropsAd polymer_props_ad(polymer_props);
//    polymer_props.reset(new PolymerPropsAd(*deck, *grid->c_grid()));
    // check_well_controls = param.getDefault("check_well_controls", false);
    // max_well_control_iterations = param.getDefault("max_well_control_iterations", 10);
    // Rock compressibility.
    // Gravity.
    gravity[2] = deck->hasField("NOGRAV") ? 0.0 : unit::gravity;
    // Init state variables (saturation and pressure).

    if (param.has("init_saturation")) {
        initStateBasic(*grid->c_grid(), *props, param, gravity[2], state);
    } else {
        initStateFromDeck(*grid->c_grid(), *props, *deck, gravity[2], state);
    }

    bool use_gravity = (gravity[0] != 0.0 || gravity[1] != 0.0 || gravity[2] != 0.0);
    const double *grav = use_gravity ? &gravity[0] : 0;

    // Linear solver.
    LinearSolverFactory linsolver(param);

    // Write parameters used for later reference.
    bool output = param.getDefault("output", true);
    std::ofstream epoch_os;
    std::string output_dir;
    if (output) {
        output_dir =
            param.getDefault("output_dir", std::string("output"));
        boost::filesystem::path fpath(output_dir);
        try {
            create_directories(fpath);
        }
        catch (...) {
            OPM_THROW(std::runtime_error, "Creating directories failed: " << fpath);
        }
        std::string filename = output_dir + "/epoch_timing.param";
        epoch_os.open(filename.c_str(), std::fstream::trunc | std::fstream::out);
        // open file to clean it. The file is appended to in SimulatorTwophase
        filename = output_dir + "/step_timing.param";
        std::fstream step_os(filename.c_str(), std::fstream::trunc | std::fstream::out);
        step_os.close();
        param.writeParam(output_dir + "/simulation.param");
    }


    std::cout << "\n\n================    Starting main simulation loop     ===============\n"
              << "                        (number of epochs: "
              << (deck->numberOfEpochs()) << ")\n\n" << std::flush;

    SimulatorReport rep;
    // With a deck, we may have more epochs etc.
//    WellState well_state;
    int step = 0;
    SimulatorTimer simtimer;
    // Use timer for last epoch to obtain total time.
    deck->setCurrentEpoch(deck->numberOfEpochs() - 1);
    simtimer.init(*deck);
    const double total_time = simtimer.totalTime();
    for (int epoch = 0; epoch < deck->numberOfEpochs(); ++epoch) {
        // Set epoch index.
        deck->setCurrentEpoch(epoch);

        // Update the timer.
        if (deck->hasField("TSTEP")) {
            simtimer.init(*deck);
        } else {
            if (epoch != 0) {
                OPM_THROW(std::runtime_error, "No TSTEP in deck for epoch " << epoch);
            }
            simtimer.init(param);
        }
        simtimer.setCurrentStepNum(step);
        simtimer.setTotalTime(total_time);

        // Report on start of epoch.
        std::cout << "\n\n--------------    Starting epoch " << epoch << "    --------------"
                  << "\n                  (number of steps: "
                  << simtimer.numSteps() - step << ")\n\n" << std::flush;

        // Create new wells, well_state
//        WellsManager wells(*deck, *grid->c_grid(), props->permeability());
        // @@@ HACK: we should really make a new well state and
        // properly transfer old well state to it every epoch,
        // since number of wells may change etc.
  //      if (epoch == 0) {
    //        well_state.init(wells.c_wells(), state);
    //    }

        // Create and run simulator.
        std::vector<double> src(grid->c_grid()->number_of_cells, 0.0);
        src[0] = 10. / Opm::unit::day;
        src[grid->c_grid()->number_of_cells-1] = -10. / Opm::unit::day;
        PolymerInflowBasic polymer_inflow(param.getDefault("poly_start_days", 300.0)*Opm::unit::day,
                                      param.getDefault("poly_end_days", 800.0)*Opm::unit::day,
                                      param.getDefault("poly_amount", polymer_props.cMax()));
        SimulatorFullyImplicitTwophasePolymer simulator(param,
                                                 *grid->c_grid(),
                                                 *new_props,
                                                 polymer_props_ad,
                                                 linsolver,
                                                 polymer_inflow,
                                                 src);
        if (epoch == 0) {
            warnIfUnusedParams(param);
        }
        SimulatorReport epoch_rep = simulator.run(simtimer, state);
        if (output) {
            epoch_rep.reportParam(epoch_os);
        }
        // Update total timing report and remember step number.
        rep += epoch_rep;
        step = simtimer.currentStepNum();
    }

    std::cout << "\n\n================    End of simulation     ===============\n\n";
    rep.report(std::cout);

    if (output) {
        std::string filename = output_dir + "/walltime.param";
        std::fstream tot_os(filename.c_str(),std::fstream::trunc | std::fstream::out);
        rep.reportParam(tot_os);
    }

}
catch (const std::exception &e) {
    std::cerr << "Program threw an exception: " << e.what() << "\n";
    throw;
}

