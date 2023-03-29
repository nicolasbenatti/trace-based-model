# Trace Based Model (TBM)

## What is TBM?

TBM is a performance simulator, designed to get performance measurements very
early in the design cycle of processors/accelerators. It provides a way for
testing the impact of new design ideas, before any detailed implementation. In
order to do hardware-software codesign, it is necessary to have a highly
configurable, easily extended performance simulator that gives detailed,
actionable feedback to hardware engineers about the bottlenecks of different
microarchitectural options; that can give detailed, actionable feedback to
software engineers about the performance of the inner loops of their code; and
that can be used to evaluate a range of different design microarchitectural
decisions. Having a configurable, and extensible, performance simulator enables
the software and hardware engineers to have the conversations that are critical
to hardware-software codesign and replaces guesswork and hunches with data.

TBMs approach is to use an existing functional simulator to generate a
functional trace and to build a “trace based model” (TBM) that models the
control/scheduling logic of the microarchitecture but does not need to calculate
the result of each instruction. The overall usage of a TBM: a functional
simulator simulates execution of a program and generates a functional trace; TBM
processes the functional trace using a microarchitectural model and produces
information about the performance of the program on the microarchitecture being
modeled. TBM is intended to be small and simple. Since it is only concerned with
implementing the control/scheduling logic, it is possible to achieve high levels
of code reuse and make it highly configurable.

## Project structure

### Prerequisite:

The make scripts expect the following environment variables and directory
structure:

* `ROOTDIR` should be set to the root directory of the sparrow source tree.
* `OUT` should be set to the directory where build artifacts should be created
  (e.g. `$ROOTDIR/out`).
* To use spike (RISC-V simulator), the spike executable should be in
  `$OUT/host/spike/bin/spike`.
* To update the RISC-V pipe-maps, the
  [riscv-opcodes](https://github.com/riscv/riscv-opcodes) repo (commit
  `6c34f60fe290613b7ba1d280b29a41179c399e69`) should be checked out at
  `$ROOTDIR/toolchain/riscv-opcodes`.

### Executables:

- `tbm/gentrace-spike.py` - reads a Spike trace and reformat it.
- `tbm/gentrace-renode.py` - reads a Renode trace and reformat it.
- `tbm/import-riscv-opcodes.py` - create and update pipe-maps.
- `tbm/merge-counters.py` - merges results from multiple runs of TBM.
- `tbm/tbm.py` - runs a trace in TBM; the main tool here.

### Python modules:

**NOTE:** currently TBM includes a single model for each of the building block
units. The intention is that other models will be added in the future to cover
uArchs that are not supported by the current models. interfaces.py defines (what
we expect to be) the API of the building blocks.

- `tbm/buffered_queue.py` - defines `Queue`, FIFO queue model.
- `tbm/counter.py` - performance counters.
- `tbm/cpu.py` - defines CPU, a cpu model (includes instances of `FetchUnit`, `SchedUnit`,
  `ExecUnit`, and `MemorySystem`).
- `tbm/disassembler.py` - bits we need to elaborate Spike traces.
- `tbm/exec_unit.py` - defines `ExecUnit`, an execution unit model (includes instances of
  `ScalarPipe`, `VectorPipe`, `scoreboard.Preemptive` and `scoreboard.VecPreemptive`).
- `tbm/fetch_unit.py` - defines `FetchUnit`.
- `tbm/functional_trace.py` - reads a trace (as generated by `gentrace-*.py`).
- `tbm/instruction.py` - defines `Instruction`, a data class representing a single
  instruction instance in the trace.
- `tbm/interfaces.py` - defines the internal API. This will be more important when we
  add different models (i.e. implementations) for the various units.
- `tbm/memory_system.py` - defines `MemorySystem`, a main memory and cache hierarchy model.
- `tbm/scalar_pipe.py` - defines `ScalarPipe`, a scalar functional unit model.
- `tbm/sched_unit.py` - defines `SchedUnit`, an issue queue model.
- `tbm/scoreboard.py` - defines `Preemptive` and `VecPreemptive`, scoreboard models.
- `tbm/tbm_options.py` - command line parsing for `tbm.py`.
- `tbm/utilities.py` - general purpose constructs.
- `tbm/vector_pipe.py` - defines `VectorPipe`, a vector functional unit model.

### TBM configuration files:

- `config/instruction.fbs` - FlatBuffer schema for the Instruction data class (used for
  saving elaborated traces). The `FBInstruction.Instruction` module is generated
  from this file.
- `config/rvv-simple.yaml` - a uArch configuration example.
- `config/uarch.schema.json` - JSON schema for uArch configuration files.
- `pipe_maps/riscv/*.json` - pipe-maps, mapping RISC-V opcodes to functional units.

### Build files:

- `Makefile` - builds things that are needed for tbm to run.
- `common.mk`
- `tbm.mk` - rules for running tbm.


## How to use the make files

### Building TBM:

Before running any of the TBM tools you must run `make all`.

To update the RISC-V pipe-maps in `pipe_maps/riscv` run `make riscv_pipe_maps`.
This will import missing opcodes from `$ROOTDIR/toolchain/riscv-opcodes`, and
will remove spurious ones. New opcodes are mapped to "UNKNOWN". You can also
update individual RISC-V pipe-maps like this: `make pipe_maps/riscv/rv32a.json`.

To run a linter on all the Python files in `tbm/` run `make lint`.

To type-check all the Python scripts run `make type-check`. After running the
type checker you can merge inferred types back to the `.py` files by running
`make merge-pyi` to merge into all `.py` files, or `make merge-pyi-MOD` to
merge into `tbm/MOD.py`.

### Running tests from sparrow

- `make -f riscv_tests.mk riscv_tests_isa` - run some of the tests from `$ROOTDIR/out/springbok/riscv-tests/isa`.
- `make -f riscv_tests.mk riscv_tests_benchmarks` - run the tests from `$ROOTDIR/out/springbok/riscv-tests/benchmarks`.
- `make -f riscv_tests.mk riscv_tests` - run the two previous targets.
- `make -f riscv_tests.mk benchmarks` - runs the above benchmarks and generates
  the file `benchmarks.md` with all the results.
- `make -f rvv_tests.mk rvv_tests` - run the tests from `$ROOTDIR/out/springbok/rvv_for_tbm/tests/`.
