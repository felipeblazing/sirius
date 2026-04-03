# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The main/default branch of this repository is `dev`.

## Project Overview

Sirius is a GPU-native SQL engine that integrates with DuckDB as an extension. It leverages NVIDIA CUDA-X libraries (cuDF, RMM) to accelerate SQL query execution on GPUs. Sirius intercepts DuckDB's physical plan execution and routes supported operations to GPU execution while gracefully falling back to DuckDB's CPU execution for unsupported cases.

**Key Integration Points:**
- DuckDB extension architecture: Sirius loads as a DuckDB extension (`sirius.duckdb_extension`)
- cuCascade: Third-party library for GPU memory management (tiered memory across GPU/host/disk)
- RAPIDS cuDF: GPU DataFrame library for data manipulation
- RMM: RAPIDS Memory Manager for GPU memory allocation

## Build System

### Environment Setup

**Using Pixi (Recommended):**
```bash
pixi shell                    # Activate environment with all dependencies
```

### Git Worktrees

When creating a new worktree, submodules are not automatically initialized. After creating the worktree, run:
```bash
git submodule update --init --recursive
```

### Building

```bash
# Full build (uses all cores by default)
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make

# If build consumes too much memory, reduce parallelism
CMAKE_BUILD_PARALLEL_LEVEL=8 make

# After build errors, clean build directory
rm -rf build
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
```

Build outputs:
- Static extension: `build/release/extension/sirius/sirius.duckdb_extension`
- Loadable extension: `build/release/extension/sirius/sirius_loadable.duckdb_extension`
- Unit test binary: `build/release/extension/sirius/test/cpp/sirius_unittest`

### Building Python API

```bash
pixi run -e duckdb-python build-duckdb-python
```

This uses a dedicated pixi environment (`duckdb-python`) with pip, pybind11, and scikit-build-core. The task automatically points `DUCKDB_SOURCE_PATH` at the repo-level `duckdb/` submodule so the Python package links against the same DuckDB version as the C++ extension.

**Usage from Python:**
```python
import duckdb

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute("LOAD 'build/release/extension/sirius/sirius.duckdb_extension'")
result = con.execute("CALL gpu_execution('SELECT ...')").fetchall()
```

## Testing

### SQL Logic Tests (End-to-End)
```bash
make test                                              # Run all SQLLogicTests
make test_debug                                        # Debug build tests

# Run specific test file
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/test/unittest --test-dir . test/sql/tpch-sirius.test
```

### C++ Unit Tests
```bash
# Build and run all unit tests
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make
build/release/extension/sirius/test/cpp/sirius_unittest

# Run tests with specific tag
build/release/extension/sirius/test/cpp/sirius_unittest "[cpu_cache]"

# Run specific test
build/release/extension/sirius/test/cpp/sirius_unittest "test_cpu_cache_basic_string_single_col"
```

Test logs are saved to: `build/release/extension/sirius/test/cpp/log`

Unit tests use Catch2 framework. Test files are in `test/cpp/` organized by component.

### Performance Testing
```bash
# Requires duckdb-python to be built
python3 test/tpch_performance/generate_test_data.py {SCALE_FACTOR}
python3 test/tpch_performance/performance_test.py {SCALE_FACTOR}
```

## Code Formatting & Linting

Sirius uses pre-commit hooks for code quality:

```bash
pre-commit run -a                    # Run all hooks on all files
pre-commit install                   # Install git hooks (runs on every commit)
```

**Code style tools:**
- C++/CUDA: clang-format (style defined in `.clang-format`)
- Python: black
- CMake: cmake-format
- Spell check: codespell (custom words in `.codespell_words`)

Configuration files:
- `.clang-format`: C++/CUDA formatting rules
- `.clang-tidy`: C++ linting rules
- `.pre-commit-config.yaml`: All pre-commit hooks

## Architecture

### Super Sirius (`gpu_execution`)

The active execution engine. Uses `namespace sirius`, entry point: `CALL gpu_execution('SELECT ...')`.

- Physical plan generator: `sirius_physical_plan_generator` (`src/planner/sirius_physical_plan_generator.cpp`)
- Operators: `sirius_physical_operator` subclasses in `src/op/` (e.g., `sirius_physical_hash_join.cpp`)
- Plan builders: `src/planner/` (e.g., `sirius_plan_filter.cpp`, `sirius_plan_aggregate.cpp`)
- Engine: `src/sirius_engine.cpp`, pipelines in `src/pipeline/`
- Interface: `src/sirius_interface.cpp` (uses `sirius_interface` class)
- Task-based execution: `src/creator/`, `src/downgrade/`, `src/op/scan/`
- Extension entry point: `src/sirius_extension.cpp`
- Expression evaluation: `src/expression_executor/`
- Runtime configuration: `src/config.cpp` / `src/include/config.hpp`
- CUDA kernels: `src/cuda/` (cuDF wrappers, expression dispatch)

> **Note:** A legacy code path (`gpu_processing`, `namespace duckdb`) still exists in `src/operator/`, `src/plan/`, `src/gpu_executor.cpp` etc. All new development targets Super Sirius.

### Super Sirius Documentation

Comprehensive documentation lives in `docs/super-sirius/` — see [README](docs/super-sirius/README.md) for index and reading order. **Read these docs before modifying Super Sirius code.**

### Logging

```bash
export SIRIUS_LOG_DIR=/path/to/logs      # Default: ${CMAKE_BINARY_DIR}/log
export SIRIUS_LOG_LEVEL=debug            # Levels: trace, debug, info, warn, error
```

## Development Guidelines

Unsupported operations (data types, operators, row counts exceeding cuDF limits) fall back to DuckDB CPU execution via `src/fallback.cpp`.

### CMake Notes

- Uses CUDA 13+ (specified in `pixi.toml` features)
- Requires C++20 and CUDA standard 20
- Separable compilation enabled for CUDA (`CMAKE_CUDA_SEPARABLE_COMPILATION ON`)
- GPU architectures: Turing through Blackwell (75, 80, 86, 90a, 100f, 120a, 120)
- Links against: cudf::cudf, rmm::rmm, libnuma, libconfig++, absl::any_invocable, spdlog, cuCascade

## Extension Development

This is a DuckDB extension project using the extension template. The build system integrates with DuckDB's extension infrastructure via `extension-ci-tools`.

**Key files for extension integration:**
- `Makefile`: Thin wrapper including `extension-ci-tools/makefiles/duckdb_extension.Makefile`
- `extension_config.cmake`: Specifies which extensions to load (sirius, json, tpcds, tpch, parquet, icu)
- `src/sirius_extension.cpp`: Extension registration (LoadInternal function)

**Extension API Usage:**

CLI:
```sql
LOAD 'build/release/extension/sirius/sirius.duckdb_extension';
CALL gpu_execution('SELECT ...');
-- Legacy mode (requires gpu_buffer_init first):
CALL gpu_buffer_init('1 GB', '2 GB');
CALL gpu_processing('SELECT ...');
```

Python (requires `pixi run -e duckdb-python build-duckdb-python` first):
```python
con = duckdb.connect('db.duckdb', config={"allow_unsigned_extensions": "true"})
con.execute("LOAD '/path/to/sirius.duckdb_extension'")
con.execute("CALL gpu_execution('SELECT ...')").fetchall()
```

## Claude Code Skills

Sirius includes Claude Code skills for performance analysis and dataset management. Invoke them via slash commands:

| Skill | Command | Description |
|-------|---------|-------------|
| Profile Analyzer | `/profile-analyzer` | Analyzes GPU performance from nsys profiles — kernel occupancy, memory bandwidth, operator attribution, and regression detection. |
| Dataset Manager | `/dataset-manager` | Manages TPC-H parquet datasets — generate at any scale factor, consolidate files, inspect layout, optimize row groups. |
| Optimization Advisor | `/optimization-advisor` | Maps GPU hotspots from nsys profiles to source functions, detects efficiency bottlenecks, sync overhead, and parallelism opportunities. |
| TPC-DS Benchmark | `/tpcds-benchmark` | Runs TPC-DS benchmarks on Legacy Sirius, Super Sirius, or DuckDB CPU baseline — generate data, execute queries, and compare results. |

**Useful debugging tools:**
- `tools/parse_pipeline_log.py`: Parses Sirius pipeline logs to show per-operator row counts for debugging incorrect query results.

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Multi-GPU Execution for Sirius**

Multi-GPU execution support for the Sirius GPU SQL engine. Enables queries to schedule pipeline tasks across multiple GPUs with data-locality-aware reservation and NUMA-aware memory downgrade. Builds on the existing single-GPU execution engine (Super Sirius / `gpu_execution`) and the cucascade memory management library.

**Core Value:** Any query can transparently execute across multiple GPUs, with tasks scheduled to GPUs where their data already resides, and memory pressure handled by downgrading to the correct NUMA domain.

### Constraints

- **Hardware variability**: Must work on any GPU/NUMA topology (1 GPU to 8+ GPUs, 1-4+ NUMA domains)
- **Backward compatibility**: Single-GPU systems must work identically to today (no regression)
- **Memory safety**: Reservation system must prevent OOM across all GPUs simultaneously
- **cucascade submodule**: Changes to cucascade are allowed but should be minimized and well-tested
- **Super Sirius only**: All changes target the `namespace sirius` / `gpu_execution` code path
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- C++ 20 - GPU processing engine, physical operators, expression execution, memory management
- CUDA 20 - GPU kernels, data operations, join/aggregate implementations (`src/cuda/`)
- Python 3.12+ - Optional: DuckDB Python bindings for testing and client applications
- Bash - Build scripts and environment setup (`scripts/`, `setup_test_datasets.sh`)
- CMake - Cross-platform build system
- SQL - Test queries and performance benchmarks (`test/sql/`, `scripts/`)
## Runtime
- Linux (x86_64, aarch64)
- NVIDIA CUDA 12.x or 13.x (configurable via pixi features)
- NVIDIA GPU drivers >= 570 (per CLAUDE.md)
- Pixi (recommended) - Conda-based environment with CUDA, compilers, dependencies
- Manual conda/conda-forge setup also supported
- Lockfile: Not explicit (uses `pixi.lock` for pixi environments)
## Frameworks
- DuckDB 1.4.4 - SQL parser, optimizer, execution framework, extension host
- cuDF (RAPIDS) 26.02.* - GPU DataFrame operations, joins, aggregates, sorting
- RMM (RAPIDS Memory Manager) - GPU memory allocation and management
- Catch2 - C++ unit test framework (bundled in `duckdb/third_party/catch`)
- CMake 4.1.* - Build configuration
- Ninja - Build backend
- clang 21-22 - C++ compiler (standard, also supports clang-release configs)
- clang-format - C++ code formatting
- clang-tidy - C++ linting
- pre-commit 2.x - Git hooks for code quality
- spdlog 1.8.* - Structured logging framework with daily file rotation
## Key Dependencies
- libcudf 26.02.* - GPU accelerated DataFrame library (RAPIDS)
- librmm * - RAPIDS Memory Manager for GPU memory allocation
- cuCascade (local submodule) - GPU memory reservation and tiered memory management
- libconfig 3.1.* - Configuration file parsing (`.cfg` file support)
- libnuma * - NUMA support for pinned memory management
- abseil-cpp 20260107.0+ - Provides `absl::any_invocable` for GPU task dispatch
- libcurand-dev * - CUDA random number generation (via CUDA toolkit)
- cuda-nvcc * - NVIDIA CUDA compiler
- cuda-nvml-dev * - CUDA profiler API access for `cudaProfilerStart/Stop()`
- spdlog 1.8.* - Structured logging
- cmake-format 0.6.13 - CMake code formatting
- codespell 2.4.1 - Spell checker for code (custom words in `.codespell_words`)
- black 25.1.0 - Python code formatter
## Configuration
- `pixi.toml` - Project dependencies and environment specification
- `CMakeLists.txt` - Main build configuration
- `extension_config.cmake` - DuckDB extension loading configuration
- `cmake/CMakePresets.json` - Build presets (release, debug, relwithdebinfo, clang variants)
- `.clang-format` - C++/CUDA formatting rules
- `.pre-commit-config.yaml` - Git hooks (clang-format, clang-tidy, black, codespell, cmake-format)
- Makefile - Thin wrapper around CMake for convenience (release, debug, test targets)
- `.cfg` file support via libconfig++ for operator parameters and memory configuration
- Environment variables for logging: `SIRIUS_LOG_LEVEL` (trace/debug/info/warn/error), `SIRIUS_LOG_DIR`
- DuckDB settings injected at runtime for Sirius configuration
## Platform Requirements
- Linux system (x86_64 or aarch64)
- NVIDIA CUDA 12.x or 13.x toolkit
- NVIDIA GPU with Turing+ architecture (CC 7.5+)
- C++ compiler: clang 21+ (via pixi)
- CMake 4.1+
- At least 16 GB GPU memory for TPC-H testing (configurable)
- At least 32 GB system RAM for full builds (parallel compilation consumes ~4GB per core)
- Linux with NVIDIA GPU drivers >= 570
- CUDA runtime libraries (bundled in extension or provided by runtime)
- DuckDB database engine (v1.4.4 or later for stability)
- GPU memory: 2-4 GB minimum for basic queries, 8+ GB recommended for analytics
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- C++ source: `snake_case.cpp` (e.g., `sirius_physical_hash_join.cpp`, `gpu_expression_translator.cpp`)
- C++ headers: `snake_case.hpp` (e.g., `sirius_interface.hpp`, `fallback.hpp`)
- CUDA kernels: `snake_case.cu` (e.g., `gpu_hash_join.cu`)
- Test files: `test_*.cpp` (e.g., `test_config.cpp`, `test_gpu_execution_tpch.cpp`)
- SQL logic tests: `*.test` (e.g., `tpch-sirius.test`, `clickbench-sirius.test`)
- Snake case: `bind_prepared_statement_parameters()`, `collect_bound_ref_indices()`, `moveDataToCPU()` (for DuckDB C++ interop functions that match DuckDB style)
- Mixed when matching DuckDB API: `sirius_process_error()` combines snake_case with prefix
- Static/utility functions in files: snake_case with descriptive names
- Class methods: snake_case (e.g., `are_conditions_supported()`, `execute()`, `get_types()`)
- Getters and setters: `get_*()`, `set_*()` pattern (e.g., `get_result()`, `get_types()`)
- Local variables: `snake_case` (e.g., `cpu_result`, `gpu_sql`, `chunk_id`)
- Member variables: `snake_case_` with trailing underscore for private members (e.g., `is_initialized_`)
- Loop variables: Single letter or abbreviated snake case (e.g., `i`, `c`, `r`, `const auto& cond`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `CPU_CACHE_TEST_MEM_SF`, `CATCH_CONFIG_RUNNER`)
- Classes: `PascalCase` (e.g., `GPUBufferManager`, `SiriusContext`, `GPUExecutionFixtureBase`)
- Enums: `PascalCase` (e.g., `MemoryBarrierType`, `TaskCreationHint`)
- Struct names: `snake_case` or `PascalCase` depending on context (e.g., `task_creation_hint`, `sirius_config_env_guard`)
- Template parameters: `PascalCase` (e.g., `TARGET`)
- Type aliases and using declarations: `snake_case` (e.g., `using TestEventListenerBase = ...`)
- Primary: `sirius` (new GPU execution path) or `duckdb` (shared/legacy code)
- Sub-namespaces: `sirius::op`, `sirius::test`, `sirius::planner`, `duckdb::*` (following DuckDB conventions)
- Namespace closing: `}  // namespace name` with comment
## Code Style
- Tool: `clang-format` configured via `.clang-format`
- Line width: 100 characters (ColumnLimit: 100)
- Indentation: 2 spaces, no tabs (TabWidth: 2, UseTab: Never)
- Brace style: WebKit (opening brace on same line as control statement)
- Pointer alignment: Left (`int* ptr` not `int *ptr`)
- `AlignAfterOpenBracket: Align` — Align function parameters/arguments
- `BreakConstructorInitializers: BeforeColon` — Colon on new line for constructor init lists
- `ConstructorInitializerAllOnOneLineOrOnePerLine: true` — No mixed init list formatting
- `AllowShortFunctionsOnASingleLine: All` — Single-line OK for short functions
- `BinPackArguments: false` — Parameters on separate lines when wrapping
- `BinPackParameters: false` — Same for function declarations
- `AlwaysBreakTemplateDeclarations: Yes` — Template declaration keywords on new line
- Tool: `clang-tidy` configured via `.clang-tidy`
- Enabled checks: `modernize-*`, `performance-*`, `clang-analyzer-*` (with specific exclusions)
- `WarningsAsErrors: '*'` — Treat warnings as errors
- Key disabled checks: `modernize-use-equals-default`, `modernize-use-trailing-return-type` (stylistic), `clang-analyzer-cplusplus.NewDeleteLeaks` (has bugs)
- clang-format: C++/CUDA code formatting (auto-fix: `-i`)
- black: Python formatting
- codespell: Spell checking with custom words in `.codespell_words`
- cmake-format: CMake file formatting
- Standard hooks: trailing whitespace, YAML/JSON checks, large files, mixed line endings
## Import Organization
- `IncludeBlocks: Regroup` — Regroup includes by priority
- `SortIncludes: true` — Sort within each group
- `SortUsingDeclarations: true` — Sort `using` statements
- Sirius includes use relative paths from `src/include`: `#include "op/sirius_physical_hash_join.hpp"`
- DuckDB includes use angle brackets with full path: `#include <duckdb/main/client_context.hpp>`
- cuDF/RAPIDS includes: angle brackets `#include <cudf/...>`
## Error Handling
- DuckDB exceptions: `throw duckdb::InternalException(...)`, `throw duckdb::InvalidInputException(...)`
- Sirius context: `throw std::runtime_error(...)` for initialization/config errors
- Assertions: `D_ASSERT(condition)` from DuckDB for debug-only checks
- Error data: `duckdb::ErrorData` struct with `FinalizeError()`, `ConvertErrorToJSON()` methods
- Visitor pattern for checking unsupported operations
- Switch on expression type/operator type with explicit case handlers
- Recursively check child expressions
- Throw with formatted message on unsupported feature
## Logging
- `SIRIUS_LOG_TRACE(...)` — Trace level
- `SIRIUS_LOG_DEBUG(...)` — Debug level
- `SIRIUS_LOG_INFO(...)` — Info level
- `SIRIUS_LOG_WARN(...)` — Warning level
- `SIRIUS_LOG_ERROR(...)` — Error level
- `SIRIUS_LOG_FATAL(...)` — Fatal (CRITICAL) level
- In `.cu` files: macros expand to no-ops (spdlog cannot be compiled by nvcc)
- Use macros in `.cpp` files only
- `SIRIUS_LOG_LEVEL`: debug, info, warn, error (defaults to Config::LOG_LEVEL)
- `SIRIUS_LOG_DIR`: Directory for log files (defaults to CMAKE_BINARY_DIR/log)
## Comments
- Algorithm explanation: Complex join logic, memory management decisions
- Performance notes: Why specific optimizations were chosen
- Warnings about gotchas: "Mixed join: cuDF requires equality and conditional columns to be disjoint"
- References to external docs: Link to DuckDB or cuDF docs for non-obvious patterns
- Not for obvious code: Don't comment `++i` or `value += 1`
- Use `/// @brief` for function descriptions (copied from RAPIDS conventions)
- Use `/// @param` for parameters
- Use `/// @return` for return values
- Example (`test/cpp/unittest.cpp`):
## Function Design
- Prefer small, focused functions (max ~50 lines for critical paths)
- Static helper functions for complex logic (e.g., `collect_bound_ref_indices()`)
- Use early returns to reduce nesting
- Use `const auto&` for loop variables when iterating containers: `for (auto const& cond : conditions)`
- Use `auto&` for mutable references
- Use `duckdb::unique_ptr<T>` for owned resources
- Use `duckdb::shared_ptr<T>` for shared ownership (GPU memory wrappers)
- DuckDB-style parameters: pass by const reference, return smart pointers
- Errors: throw exceptions (DuckDB pattern)
- Success: return value via `duckdb::unique_ptr<T>` or by-reference parameter
- Optional: use `std::optional<T>` (e.g., float_tolerance parameter in tests)
## Module Design
- Header files contain declarations; implementations in `.cpp`
- Public APIs in headers under `src/include/`; implementation details in `src/`
- Use `inline` for small utility functions in headers
- Not used; direct includes by path (e.g., `#include "op/sirius_physical_hash_join.hpp"`)
- Public: Constructors, main methods, public getters
- Protected: Virtual methods, protected data for derived classes
- Private: Implementation details, private members with trailing `_`
## Type Safety
- `duckdb::unique_ptr<T>` — Sole ownership (RAII)
- `duckdb::shared_ptr<T>` — Shared ownership
- `duckdb::optional_ptr<T>` — Nullable non-owning pointer (replaces `T*` in DuckDB)
- `expr.Cast<TargetType>()` — DuckDB-safe cast for expressions
- `dynamic_cast<T*>` — General C++ polymorphism (avoid in hot paths)
- `reinterpret_cast<T*>` — Only for GPU memory pointers
- Mark member functions `const` if they don't modify state
- Use `const auto&` in loops: `for (auto const& item : items)`
- Use `const` on parameters that shouldn't be modified
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Dual-mode execution: Legacy Sirius (`gpu_processing`, namespace `duckdb`) and New Sirius (`gpu_execution`, namespace `sirius`)
- Pipeline-based parallel execution with stream-per-thread GPU scheduling
- Three-tier memory management (GPU/host/disk) via cuCascade integration
- Task-driven architecture with dynamic task creation, GPU pipeline execution, and memory downgrading
## Layers
- Purpose: Bridge between DuckDB's query execution and Sirius GPU engine
- Location: `src/sirius_extension.cpp`, `src/include/sirius_interface.hpp`, `src/sirius_interface.cpp`
- Contains: Extension registration, table function bindings (`gpu_processing`, `gpu_execution`), query result management
- Depends on: DuckDB core API, physical plan interfaces
- Used by: DuckDB query executor
- Purpose: Convert DuckDB's logical operator trees to Sirius-specific physical operators
- Location: `src/planner/` (new), `src/include/planner/`
- Contains: `sirius_physical_plan_generator`, plan builders for each operator type (`sirius_plan_filter.cpp`, `sirius_plan_aggregate.cpp`, etc.)
- Depends on: DuckDB LogicalOperator interface, operator type definitions
- Used by: Extension layer to prepare executable plans
- Purpose: Orchestrate query execution across thread coordinator, task creator, scan executor, pipeline executor, and downgrade executor
- Location: `src/sirius_engine.cpp`, `src/include/sirius_engine.hpp`
- Contains: `sirius_engine` (orchestrator), pipeline collection and scheduling, operator ID management
- Depends on: Physical operators, pipelines, repositories, memory management
- Used by: Interface layer to execute prepared plans
- Purpose: Manage operator chains as independent parallel execution units
- Location: `src/pipeline/`, `src/include/pipeline/`
- Contains: `sirius_pipeline`, `sirius_meta_pipeline`, pipeline task states, GPU task queues, execution handlers
- Depends on: Physical operators, data repositories, CUDA execution
- Used by: Engine, task creator, GPU executors
- Purpose: GPU-accelerated implementations of query operators
- Location: `src/op/` (new Sirius), `src/include/op/`
- Contains: Scan operators (`sirius_physical_table_scan`, `sirius_physical_parquet_scan`), compute operators (filter, aggregate, join), merge operators
- Depends on: cuDF libraries, expression executor, data representations
- Used by: Pipeline executor
- Purpose: Manage creation and scheduling of GPU-bound and scan tasks
- Location: `src/creator/` (task creation), `src/downgrade/` (memory downgrade), `src/op/scan/` (scan tasks), `src/pipeline/` (GPU pipeline tasks)
- Contains: `task_creator` (schedules pipeline tasks), `downgrade_executor` (moves data across memory tiers), scan task implementations
- Depends on: Operator data, repositories, memory reservations, thread pools
- Used by: Engine during execution
- Purpose: Evaluate SQL expressions on GPU
- Location: `src/expression_executor/`, `src/include/expression_executor/`, `src/cuda/expression_executor/`
- Contains: Expression translator (SQL AST → GPU code), dispatcher, GPU kernels for comparison, string ops, materialization
- Depends on: cuDF, DuckDB expression AST
- Used by: Filter, projection, join operators
- Purpose: GPU kernels and cuDF wrappers
- Location: `src/cuda/`, `src/cuda/cudf/`, `src/cuda/operator/`
- Contains: cuDF wrappers (aggregate, join, orderby, groupby), operator-specific kernels, utilities
- Depends on: RAPIDS cuDF, RMM, CUDA runtime
- Used by: Physical operators
- Purpose: GPU memory allocation, caching, spilling with three-tier hierarchy
- Location: `src/memory/`, `src/include/memory/`, `cucascade/`
- Contains: `sirius_memory_reservation_manager` (memory leases), cuCascade data repositories and memory spaces
- Depends on: cuCascade library, CUDA runtime
- Used by: All layers for GPU resource allocation
## Data Flow
- **Pipeline State**: `sirius_pipeline` tracks source, operators, sink; maintains dependencies via `sirius_meta_pipeline`
- **Task State**: `gpu_pipeline_task_local_state` holds input data, reservation, retry count; `gpu_pipeline_task_global_state` shared across tasks
- **Data State**: `data_repository` manages batches across tiers; operator outputs are batches (cudf::table or spilling allocation)
- **Memory State**: `sirius_memory_reservation_manager` tracks leases; `downgrade_executor` coordinates tier movement
## Key Abstractions
- Purpose: Base class for all GPU-executable query operators
- Examples: `sirius_physical_filter`, `sirius_physical_hash_join`, `sirius_physical_grouped_aggregate`, `sirius_physical_table_scan`
- Pattern: Virtual `Execute()` and `Finalize()` methods; children stored as `vector<unique_ptr<sirius_physical_operator>>`; sink/source state management
- Location: `src/include/op/sirius_physical_operator.hpp`, `src/op/`
- Purpose: Chain of operators executed as a unit
- Examples: Filter pipeline, join build pipeline, aggregate pipeline
- Pattern: Contains source operator, operator vector, sink operator; manages dependencies via `sirius_meta_pipeline`; scheduled independently
- Location: `src/include/pipeline/sirius_pipeline.hpp`, `src/pipeline/`
- Purpose: Executable work unit: pipeline + input data + memory reservation
- Pattern: Encapsulates pipeline reference, input batch (via `operator_data`), retry context, memory reservation; executed by GPU thread
- Location: `src/include/pipeline/gpu_pipeline_task.hpp`, `src/pipeline/gpu_pipeline_task.cpp`
- Purpose: Container for operator output batches with automatic tier management
- Pattern: Created by engine per operator; stores `shared_ptr<data_batch>` entries; queries memory pressure; supports migration callbacks
- Location: Integrated from `cucascade/` (third-party), used throughout execution
- Purpose: Wrapper for vector of data batches passed between operators
- Examples: `operator_data`, `partitioned_operator_data`
- Pattern: Holds `vector<shared_ptr<data_batch>>`; provides const access interface
- Location: `src/include/op/sirius_physical_operator.hpp`
- Purpose: Lease on GPU memory to prevent oversubscription
- Pattern: Allocated by `sirius_memory_reservation_manager` before task execution; released after task completion
- Location: `src/include/memory/sirius_memory_reservation_manager.hpp`
## Entry Points
- Location: `src/sirius_extension.cpp`, function registered as DuckDB table function
- Triggers: User calls `CALL gpu_execution('SELECT ...')`
- Responsibilities: Parse SQL, prepare statement, create `sirius_interface`, run query, return results
- Location: `src/include/sirius_engine.hpp`, `src/sirius_engine.cpp`
- Triggers: Called after engine initialization with physical plan
- Responsibilities: Spawn task creator and scan executor threads, monitor execution completion, aggregate results
- Location: `src/include/creator/task_creator.hpp`, `src/creator/task_creator.cpp`
- Triggers: Called by engine after thread pool started
- Responsibilities: Poll repositories, create GPU tasks when data and memory available, submit to executor
- Location: `src/include/pipeline/gpu_pipeline_executor.hpp`, `src/pipeline/gpu_pipeline_executor.cpp`
- Triggers: GPU thread pool pulls task from queue
- Responsibilities: Execute operator chain on GPU, handle errors, trigger completion handler
## Error Handling
- **Expression Validation**: `gpu_expression_translator` checks expression support; throws `NotImplementedException` for unsupported ops (window functions, complex regex)
- **Memory Pressure**: `OomRescheduleException` allows task retry with reduced input size; `downgrade_executor` automatically migrates data
- **Data Type Checking**: Plan generation validates supported types (INTEGER, BIGINT, FLOAT, DOUBLE, VARCHAR, DATE, TIMESTAMP, DECIMAL); falls back to CPU for nested types
- **Operator Fallback**: Result collector or pipeline error triggers CPU re-execution of unsupported subtree
## Cross-Cutting Concerns
- spdlog framework, configurable via `SIRIUS_LOG_LEVEL` and `SIRIUS_LOG_DIR`
- Logged at: Physical plan generation, operator execution, task scheduling, memory allocation
- Location: `src/include/log/logging.hpp`
- Operator trees verified via `sirius_physical_operator::verify()`
- Expression translator validates AST structure before GPU code generation
- Type checking in plan generation layer
- Not applicable (GPU execution engine layer; auth handled by DuckDB)
- NVIDIA CUDA Profiler API hooks in `sirius_extension.cpp` (`cudaProfilerStart`/`cudaProfilerStop`)
- NVTX markers in pipeline execution and operator dispatch for nsys profiling
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
