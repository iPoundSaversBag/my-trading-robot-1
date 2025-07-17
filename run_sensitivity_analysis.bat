@echo OFF
REM ==============================================================================
REM
REM                      SENSITIVITY ANALYSIS RUNNER
REM
REM ==============================================================================
REM
REM FILE: run_sensitivity_analysis.bat
REM
REM PURPOSE:
REM   This batch script is designed to automate the process of running sensitivity
REM   analysis on the trading strategy. It systematically executes the main
REM   backtesting script (`ichimoku_backtest.py`) across a range of different
REM   optimization intensity levels.
REM
REM METHODOLOGY:
REM   The script iterates through a predefined list of intensity levels (1, 2, 3, 4).
REM   For each level, it calls the backtester, passing the intensity as a
REM   command-line argument. This allows for a comprehensive analysis of how
REM   the strategy's performance changes as more computational effort is spent
REM   on the optimization process.
REM
REM USAGE:
REM   Simply double-click this file or run it from the command line. It will
REM   execute the backtester four times, once for each intensity level, and
REM   the results will be saved in separate, timestamped directories inside the
REM   `plots_output` folder.
REM
REM ==============================================================================

REM Activate the virtual environment
call venv\Scripts\activate.bat

ECHO Starting Sensitivity Analysis for min_trades...

REM Define the range of min_trades values to test
SET min_trades_values=5 10 15 20 25 30

REM --- DYNAMICALLY COUNT THE NUMBER OF RUNS ---
SET /A num_runs=0
FOR %%T IN (%min_trades_values%) DO (
    SET /A num_runs+=1
)
REM Add a buffer just in case there are other recent runs you want to keep
SET /A runs_to_keep=%num_runs% + 5 

ECHO.
ECHO Running %num_runs% backtests for the following min_trades values: %min_trades_values%
ECHO Housekeeping will be set to keep the last %runs_to_keep% runs to prevent data loss.
ECHO This may take a long time.
ECHO.

FOR %%T IN (%min_trades_values%) DO (
    ECHO ======================================================
    ECHO Running analysis with --min-trades %%T
    ECHO ======================================================
    python ichimoku_backtest.py --intensity 1 --min-trades %%T --runs-to-keep %runs_to_keep%
    IF ERRORLEVEL 1 (
        ECHO.
        ECHO **************************************************
        ECHO * SCRIPT FAILED for min_trades=%%T. Halting analysis. *
        ECHO **************************************************
        GOTO:EOF
    )
)

ECHO.
ECHO ======================================================
ECHO Sensitivity Analysis Complete.
ECHO Check the 'plots_output' directory for results from each run.
ECHO ======================================================
