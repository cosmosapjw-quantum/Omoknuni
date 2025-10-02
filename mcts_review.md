**  concise review **

* Critical Issues

  - Policy loss calls F.cross_entropy with float probability targets, so the first batch will throw RuntimeError: expected scalar type Long and stop training. Use log-softmax + KL or
  convert MCTS visits to class indices before calling cross-entropy. See src/training/trainer.py:574, src/training/trainer.py:600.
  - TrainingConfig omits every field required for self-play batching (mcts_threads, batch_size_min, batch_size_max, inference_timeout_ms), yet _initialize_components reads them
  unguarded. Instantiating a loop from defaults hits AttributeError before the first iteration. Fill these fields in the dataclass (or gate their usage). References: src/training/
  training_loop.py:47, src/training/training_loop.py:199.
  - create_training_loop builds a flattened dict containing unknown keys (mcts_threads, etc.) and hands it straight to TrainingConfig(**flat_config), which raises TypeError. Either
  extend TrainingConfig or split the extra sections before constructing the dataclass. See src/training/training_loop.py:789.
  - SearchCoordinator.stop is defined twice; the second (decorated) definition overrides the first, so we never cancel active futures or clear active_searches, leaving threads
  running underneath shutdown. Consolidate the logic so that cancellation, queue draining, and telemetry cleanup all execute. References: src/core/search_coordinator.py:185, src/core/
  search_coordinator.py:549.

*  Major Issues

  - TrainingLoop.__init__ installs SIGINT/SIGTERM handlers unconditionally. If the loop is created off the main thread (very common in orchestration/tests) Python raises ValueError.
  Protect the registration with threading.current_thread() is threading.main_thread(). See src/training/training_loop.py:162.
  - Evaluation is effectively a stub: win_rate is hard-coded to 0.5, so checkpoints never reflect real strength deltas. Wire the outcome calculation into these metrics, or at least
  mark the placeholder to avoid shipping meaningless numbers. src/training/training_loop.py:482.
  - Experience buffer writes and samples re-load the full Parquet file on every add/read (pq.read_table inside tight loops). That turns O(N) into O(total_examples) per batch,
  making the system choke long before 1M samples. Cache a ParquetFile reader and append by row group instead. References: src/training/experience_buffer.py:236, src/training/
  experience_buffer.py:331.
  - Search scheduling oversubscribes threads badly: each coordinator worker spins up an AlphaZeroMCTS with its own ThreadPoolExecutor sized to max_threads, so N concurrent searches
  spawn N×max_threads simulation threads. The host will thrash under load. Consider reusing a shared worker pool or cutting num_threads to 1 when running inside a coordinator thread.
  See src/core/mcts.py:197, src/core/search_coordinator.py:312.

* Moderate Observations

  - Default model_path concatenation adds '/latest.pth' to models/latest.pth, yielding models/latest.pth/latest.pth; it works but creates a confusing “file-named directory” and breaks
  expectations when model_dir is absent. src/training/training_loop.py:792.
  - Evaluation/self-play reuse the same generator instance without snapshotting model weights; concurrent updates risk mixing policies mid-iteration. Ensure update_model blocks on
  outstanding games. src/training/self_play.py:342.
  - Telemetry collectors are constructed but never started, so GPU/system metrics stay at zero. Either call start_collection() or remove the unused collector. src/telemetry/
  metrics.py:65, src/training/training_loop.py:150.

* Open Questions

  - Does alphazero_py.GameState.get_current_player() return 0/1 or 1/2 post-terminal? GameStateWrapper.get_result and MCTS root initialization rely on the exact convention; worth
  double-checking with the C++ bindings. src/games/game_state.py:65, src/core/mcts.py:171.
  - Should the self-play temperature/value targets support rule variants like Renju ties beyond simple alternating players? Current outcome back-filling assumes two-player alternating
  turns only. src/training/self_play.py:440.

* Summary

  - Training cannot currently run end-to-end: fix the trainer loss, configuration schema, and coordinator shutdown before anything else.
  - Once the pipeline boots, address evaluation fidelity and replay-buffer performance to prevent silent regressions and throughput collapse.

-----
-----
detailed review is provided below
-----
-----

Omoknuni Codebase Critical Review
Overview and Context
Omoknuni is an ambitious AlphaZero-style engine targeting Gomoku/Omok, Chess, and Go on consumer
hardware (Ryzen 5900X CPU + RTX 3060 Ti GPU). The design centers on a hybrid Python/C++ approach:
Python orchestrates training and high-level logic, while performance-critical components (MCTS, game
state, neural inference) are implemented in C++ with PyBind11 for speed . The stated performance
goal is 30,000–40,000 simulations/sec including neural network (NN) inference, with ~80–92% GPU
utilization on the RTX 3060 Ti . Currently, however, the actual MCTS performance falls drastically
short – roughly 1,000 simulations/sec in tests – indicating serious inefficiencies that must be addressed
before production deployment.
Below, we critically examine the MCTS implementation, identify bottlenecks and potential bugs, and
provide ruthlessly honest recommendations for optimization. We also glean insights from documentation
to understand the intended architecture vs. the reality of the current code. The goal is to pinpoint why the
code isn’t meeting its performance targets and how to improve it. All observations are grounded in code
and documentation evidence, with an eye toward achieving a robust, production-ready alpha release.
1. MCTS Implementation: Design vs. Reality
High-Level Design: The MCTS engine ( AlphaZeroMCTS in src/core/mcts.py ) is intended as a “High-
Performance MCTS Engine with C++ Backend” . Key design features include: a C++ “structure-of-arrays”
tree for memory/cache efficiency, AVX2-vectorized PUCT selection for fast child selection, thread-safe
atomic updates with virtual loss for multi-threading, and asynchronous NN inference integration .
In theory, multiple threads should traverse a shared search tree concurrently (up to 8–12 threads) using
virtual loss to avoid collisions, thereby achieving the desired throughput . The Python layer is meant
to be minimal – ideally only coordinating search requests – with all “hot loops” in C++ or Cython (releasing
the GIL) .
Current Implementation & Performance: In practice, the current MCTS implementation does not yet
realize these ideals – it suffers from significant Python overhead and a critical logic bug, resulting in an
order-of-magnitude slowdown. The core search loop is written in Python ( AlphaZeroMCTS.search ) and
iterates for each simulation, calling into C++ for certain operations but still doing substantial work in Python
. Specifically, each simulation involves Python-level logic for game state cloning and move
application on each tree traversal step . This is problematic because it violates the stated design
principle that “Python never touches hot loops” – instead, Python is driving the most performance-critical
loop. The Global Interpreter Lock (GIL) further exacerbates this: since the PyBind11 C++ methods are not
explicitly releasing the GIL (no py::gil_scoped_release in bindings) , multi-threading within
one search cannot fully exploit multiple CPU cores. Essentially, the current code is running single-threaded
MCTS bound by Python’s speed, which explains the ~1k sims/sec ceiling.
Critical Bug – Move Application/Cloning: A serious bug in the simulation loop likely renders it not only
slow but also incorrect. The MCTS _run_simulation method clones and applies moves as follows:
# mcts.py::_run_simulation (excerpt)
new_state = current_state.clone()
result = new_state.make_move(move)
# Comment says make_move modifies in-place and returns None
current_state = new_state 【3†L391-L399】
The assumption in the comment is that make_move will modify new_state in-place and return None
. However, the actual GameState implementation contradicts this. The
GameStateWrapper.make_move method creates a new clone internally and returns a new state
wrapper instead of modifying the original (it clones self._state again and applies the move) . Thus,
in _run_simulation the result is a new state with the move applied, but current_state is set to
new_state – which still represents the old state without the move. In short, the code is ignoring the
applied move and continuing the simulation with an unmodified state. This bug can break the tree
traversal (likely causing repeated expansion of the same node or other inconsistencies) and definitely
wastes computation by cloning twice per move. It also explains poor simulation throughput: each
simulation is doing roughly double the cloning work and not advancing game state properly. Fix: Use the
intended in-place move logic. For example, calling current_state.apply_move_inplace(move) on the
clone (or adjusting make_move to modify and return self) would correct the logic and halve the overhead
immediately. This bug slipped through because the project’s tests use a Python SingleThreadedMCTS
mock rather than the real engine, so this core logic wasn’t exercised under real conditions.
Excess Python Overhead: Beyond the bug, even a correct Python-loop approach will struggle to hit tens of
thousands of simulations per second. Each simulation involves: (a) multiple C++ calls (for PUCT selection
and tree updates), (b) Python branching and loop overhead, and (c) interacting with game state. The game
state is backed by a C++ implementation (e.g. alphazero_py.GomokuState ), which is good for move
generation speed. However, each call to game_state.clone() and game_state.make_move() from
Python still incurs cross-language call overhead and some Python logic. The design does provide a faster
alternative – e.g. GameStateWrapper.apply_move_inplace and undo_move for potentially keeping a
single state and rolling moves forward/backward – but the current MCTS loop doesn’t leverage
these. Instead, it clones the state at each step. Even if the double-clone bug is fixed, that means one new
state allocation per tree level per simulation. For a branching factor of, say, ~150 and depth ~10, a single
simulation might clone ~10 states. At 1000 sims/sec, that’s 10,000 state copies/sec – heavy for Python to
orchestrate.
Move Mapping Overhead: Another Python-side cost is how child moves are tracked. The code uses a
Python dict _move_mapping to map each child node index to the move that led there . This
mapping is cleared and rebuilt every search. While functional, this is a potential memory and speed concern
if the tree grows large (e.g. millions of nodes). In the worst case, 10 million nodes would mean 10 million
dictionary entries in Python – an enormous overhead (likely far more memory than the compact C++ tree
itself). This undermines the memory efficiency of the SoA tree (target ~27 bytes/node) because each Python
dict entry can be dozens of bytes. It also adds overhead in lookups during simulation. Suggestion: Maintain
move-to-child associations in C++ or in a contiguous Python structure. For example, an array parallel to the
node arrays could store the move index for each node, avoiding Python dictionaries. The project already
optimizes child storage in C++ (children are allocated in contiguous blocks) – thus the i -th child of a node
corresponds to the i -th legal move in the expansion. The engine could exploit that: store the list of legal
moves at expansion time and derive moves by offset instead of per-node mapping. At minimum, replacing
the dict with a preallocated numpy array (of length max_nodes ) for moves would drastically cut Python
overhead for move lookup. This kind of low-level optimization is necessary when scaling to millions of
nodes.
Lack of True Multi-Threading: The codebase includes sophisticated support for multi-threaded MCTS
(virtual loss coordination, atomic updates, a thread pool in SearchCoordinator ), but in the current
state all simulations for a given search run on one thread. The AlphaZeroMCTS.search() method
simply runs a loop over range(simulations) in Python ; it does not spawn worker threads to divide
the simulations. The SearchCoordinator class is designed to manage multiple concurrent search tasks,
not to parallelize one search across threads (it submits each search request as a separate thread pool job)
. This means if you request one large MCTS search (e.g. 1600 simulations for a single position), it will
still execute on a single Python thread. The documentation, however, implies the intent was to have all
threads share one tree (“one tree per search: all threads share the same tree with atomics” ). To
realize that, the code would need to launch multiple threads on the same AlphaZeroMCTS instance for a
single search. For example, splitting the simulation count among threads or simply having each thread call
_run_simulation() in a tight loop on the shared tree. That is not implemented – likely due to the GIL
and Python control flow issues discussed. As a result, the current code uses at most 1 CPU core per search,
leaving tremendous performance on the table. Achieving 30k+ sims/sec will require enabling true multi-
threaded search. This could be done by moving the main loop into C++ (where threads can be spawned
without GIL contention) or by releasing the GIL in long-running C++ methods so that Python threads can
execute in parallel. For instance, a C++ method that performs, say, 100 simulations in a batch with
gil_scoped_release could allow parallel execution of those batches on multiple Python threads. Right
now, none of the PyBind11 bindings use call guards to release the GIL , meaning each C++ function
call (selection, backup, etc.) still holds the GIL. This negates much of the thread-safety scaffolding already
built (virtual loss, atomic ops). In summary: multi-thread parallelism on a single search is effectively
disabled in the alpha code. Enabling it is paramount to reach the stated throughput.
Tree Memory and Clearing: The MCTS tree implementation ( MCTSTree ) is quite memory-efficient and
scalable: ~32–40 bytes per node, support for tens of millions of nodes, and use of contiguous arrays for
cache efficiency . However, we note a potential performance issue in how the tree is reused. Each call
to AlphaZeroMCTS.search() does tree.clear() which memsets or reinitializes the entire
allocated memory pool for reuse . By default, max_tree_size is 10 million nodes (270MB) .
Clearing that means writing ~270MB of zeros – which can take on the order of hundreds of milliseconds. If
searches are repeated frequently (e.g. each move in a game or multiple searches per second during self-
play), this becomes a fixed overhead. If the tree is not fully utilized, clearing the whole space is wasteful. A
more optimal approach would be to maintain a current node count and only zero out or reset the part of
arrays actually used, or implement a custom allocator that simply resets a pointer without touching
memory (the data doesn’t need to be zeroed for correctness if node_count is reset and new nodes overwrite
old data). This is a lower priority than the Python issues, but still worth noting as the code scales up.
Reducing memory initialization overhead will help approach the upper performance bound, especially on
large max_nodes settings.
PUCT Selection and Backup: On a positive note, the C++ implementation of PUCT selection and value
backup is well-optimized. The selection uses AVX2 intrinsics to evaluate 8 children at a time and
falls back to scalar for remainder, which should significantly speed up each node traversal. The backup logic
updates visit counts and total values with atomic operations and alternates the value sign at each level
(negating as it moves up the tree) to handle perspective flipping . The backup manager appears
correct in propagating values and clipping them to [-1,1] if enabled. These components likely perform as
expected in isolation. The bottleneck is not in the mathematical operations themselves – those are fast
in C++ – but in how often they’re invoked (too many Python loops and state copies per simulation) and the
lack of parallelization. In single-thread mode, the AVX2 selection might slightly mitigate Python’s slowness,
but the impact is limited if we’re only doing 1000 sims/sec; the CPU is mostly idle. It’s worth verifying that
AVX2 is actually enabled at runtime; the code checks CPU support and can disable SIMD if not available
. Assuming a modern CPU (5900X) it should be on.
Correctness and Outcome Quality: Another aspect to scrutinize is whether the MCTS is even yielding
correct search distributions given the bug. If moves aren’t applied properly during simulations, the tree’s
structure and node values could be wrong. This might not show up in simple tests (especially since the test
environment uses a different MCTS), but in actual self-play games it could lead to suboptimal or random
moves. Fixing the move application bug is thus critical not just for speed but for search correctness. Once
fixed, the search algorithmic logic (PUCT with FPU=0, virtual loss, Dirichlet noise injection, etc.) seems solid
and in line with AlphaZero. We might consider a minor calibration: the constant fpu_value = 0.0 (first-
play urgency) and use_fpu=True are set in code , which is fine (AlphaZero often uses zero or a
slight bias for FPU). The c_puct=1.25 default is standard . These parameters can be tuned later; they
likely aren’t the cause of the low performance.
In summary, the MCTS implementation needs significant optimization work. The core data structures
and math are capable, but the integration is inefficient. The top priorities are:
Fix the state clone/move bug so simulations traverse distinct game states properly. This will
immediately improve both speed and correctness .
Eliminate Python from the inner loop. Options include moving the entire search() loop into C+
+ or Cython (ensuring the GIL is released), or at least batching simulations in C++ to amortize Python
call overhead. The design documentation even suggests a Cython-based loop example (with
nogil ) ; implementing something akin to that would align the code with its original vision.
Enable multi-threaded tree search on a single MCTS instance. This likely means spawning threads
at the C++ level (since Python threads will contend on the GIL). Given that the tree, selection, and
backup are already thread-safe by design, this is a matter of engineering the concurrent execution.
For instance, one could implement a C++ method MCTSTree::search_parallel(int
simulations, int num_threads) that distributes simulation work across threads internally,
each thread calling the existing _run_simulation logic (but written in C++). This would fully utilize
the 12-core CPU for one search, multiplying sims/sec proportionally (and also generating more
concurrent inference requests to feed the GPU).
Reduce per-simulation overhead by using in-place moves or state reuse. Ideally, avoid cloning the
entire game state for every step. A single state can be reused along each simulation path by doing
moves forward and then undoing them when backtracking (the code already has undo_move()
implemented ). This is more complex to implement but can drastically cut down object
allocations. Even without full undo logic, using apply_move_inplace on a cloned state (instead of
clone+clone for every move) will help.
Optimize move mapping to remove Python dict overhead in large searches. A C++ field for moves
or a fixed array mapping node indices to moves (at least for expanded nodes) would be much more
efficient. This change would both save memory and improve speed when iterating over children.
By addressing these points, the MCTS engine can move closer to the target performance. The difference is
stark: currently ~1k sims/sec on one thread vs. an expected ~30k on 8–12 threads. The code has the pieces
to bridge this gap; it’s a matter of implementation and debugging effort.
2. Documentation vs. Implementation
Target Hardware & Performance Goals: The documentation (README and the mcts_guide.md ) clearly
outlines the hardware and performance context. The engine is optimized for a Ryzen 9 5900X (12 cores)
and an RTX 3060 Ti GPU . This choice explains the emphasis on multi-threading and batching: with
12 CPU cores, the aim was to run 8–12 MCTS threads in parallel, and with an 8GB GPU, to batch around 32–
64 state evaluations at a time . The documentation’s performance table sets a realistic goal of 30–40k
simulations/sec with those resources . It also notes that single-GPU utilization above ~90% is unrealistic
in this setup, highlighting a pragmatic understanding of bottlenecks . Right now, the
implementation isn’t harnessing the hardware’s potential due to the single-thread limitation – as a result,
CPU utilization is low (likely one core pegged at 100%, the others idle) and GPU utilization is also low (the
inference is stubbed or trivial, so the GPU mostly waits). Bridging this gap will involve fulfilling the design
promises of parallel CPU search and asynchronous batching.
Code Infrastructure & Design Completeness: One striking aspect is the breadth of the codebase – it’s
not just an MCTS loop and a model, but a whole training ecosystem. There are modules for telemetry
( src/telemetry ), configuration, a detailed SearchCoordinator for asynchronous search requests,
and even tuning scripts (for threads, virtual loss magnitude, batch size, etc.) . The README checklist
shows dozens of tasks checked off, from basic pipeline setup to advanced features like OOM recovery, ELO
evaluation, and error handling . This indicates the project has tried to implement a “production-
ready” training system in one go. However, some of these features appear either unfinished or
unintegrated in the current alpha:
Neural Network Inference Integration: The SearchCoordinator and GPUInferenceWorker
classes suggest a robust solution for batched GPU inference (with dynamic micro-batching, timeouts
≤3ms, pinned memory, mixed precision, etc.) . Yet, in the current alpha, the inference is
essentially mocked/simulated. The SearchCoordinator._process_inference_request
simply sleeps 1ms and returns a random policy/value . The GPU worker thread is initialized
but not actually used to perform inference – _process_inference_request does not invoke
GPUInferenceWorker at all, it bypasses it. This is clearly temporary (likely done to test the search
pipeline without having a trained model in place). It means that actual GPU utilization and batching
behavior have not been tested. When moving to a real model, there will likely be integration bugs or
performance tuning needed (ensuring that the coordinator properly enqueues requests to the GPU
worker, that results come back correctly, etc.). In short, the GPU inference pipeline is present in
code but not activated. This should be completed as part of continuing development. Otherwise
the engine, even if the MCTS core becomes fast, will be bottlenecked by inference.
Thread Pool Usage: The SearchCoordinator uses a ThreadPoolExecutor to handle multiple
search requests concurrently . This is great for self-play (playing many games in parallel, each
game’s moves evaluated by a separate thread). However, combining this with multi-thread within
each search is tricky – one must ensure not to oversubscribe CPU cores or create contention. The
tuning scripts (e.g. tune_threads.py ) indicate the intention to find an optimal balance, likely
around 8 threads for search given 12 cores . The code as written can either run 8 separate
searches or one search with 8 threads, but not the latter yet. Once intra-search threading is enabled,
developers should coordinate it with the SearchCoordinator (perhaps run single-thread
searches when many different games are in flight, but use all threads when evaluating a single
critical position, etc.). The infrastructure to do this scheduling isn’t fully there – currently
SearchCoordinator assumes each search uses one thread. This is a design decision to revisit.
Since the documentation emphasizes “shared-tree MCTS” and not running many independent trees
for one position , it might be wise to implement multi-threading inside
AlphaZeroMCTS.search and possibly use fewer parallel search requests at once to avoid
thrashing. These details will need careful attention and likely empirical tuning.
Testing and Development Status: The project appears to be in an alpha stage, meaning core
features are in place but not fully validated. The README claims “Alpha Release (v1.0.0-alpha) – all 53
implementation tasks completed” , but the reality is that some of those tasks are only superficially
completed. For example, T014: GPU inference worker is implemented in code, but as noted, it isn’t
actually wired into the running pipeline yet. The contract tests and integration tests often rely on
mocks or simplified conditions. For instance, the MCTS integration test uses a
SingleThreadedMCTS Python class with a MockMCTSTree , rather than the actual C++
AlphaZeroMCTS . This allowed testing the conceptual MCTS cycle without needing the
compiled extension, but it also meant the real implementation wasn’t end-to-end tested. As a result,
issues like the move application bug or performance shortfalls weren’t caught by tests. The
performance test in test_mcts_single_thread.py simply checks that 1000 simulations
execute faster than 0.5s (>2000 sims/sec) as a very lenient threshold for the mock setup – it’s
essentially ensuring the loop isn’t egregiously slow, but it’s far below the 10k target mentioned in
comments . In summary, the testing so far has been modest, focusing on functional
correctness with mocks rather than true performance on real components. Moving forward,
rigorous performance profiling and real-game testing will be crucial.
Complexity vs. Stability: The codebase’s breadth (telemetry, error handling, fallback mechanisms,
etc.) is impressive but could prove brittle. There are custom error types and monitors
( ThreadHealthMonitor , error_reporter , etc.) to catch and respond to failures in threads .
While important for a long-running training process, these add complexity that could themselves
introduce bugs (e.g., error handling not properly synchronized or false positives causing an
emergency shutdown). It might be prudent to temporarily simplify some of these systems when
focusing on core performance – or at least ensure they are well tested under high load. In an alpha
release, it’s often better to get the core algorithms solid and then layer on robustness features. For
example, one could disable or simplify the telemetry and focus on raw throughput first (the
telemetry indicates metrics like searches per second, queue depth, etc., which could interfere or
slightly slow the system; albeit likely minor overhead).
Documentation Accuracy: Notably, the documentation and code sometimes diverge. The
mcts_guide.md describes an alternate implementation approach using Cython with an example function
for selection , which is not actually used (the project opted for a pure C++ approach). It also emphasizes
releasing the GIL in C++/Cython loops , which, as discussed, hasn’t been done yet. The guide’s
performance assumptions (e.g. 10k nodes/sec single-thread target ) are unmet by an order of
magnitude in the current build. This gap suggests that either further optimizations were deferred or that
certain design assumptions didn’t pan out in practice. For instance, maybe the overhead of PyBind11 calls
was higher than expected, or the cost of repeatedly initializing large arrays (for the tree) was
underestimated. It’s important to reconcile these differences. The team should perform detailed profiling
(with tools like Python cProfile for Python overhead, and VTune or perf for C++ if needed) to pinpoint where
the time is going per simulation. That will validate or refute assumptions in the design docs.
On the positive side, the documentation provides a clear blueprint of what needs to be achieved. The
architecture diagram and “key design principles” in the guide are spot-on . To align the
implementation with these:
Principle “Python coordinates, C++ computes”: Achieved partially (C++ does compute selection and
backup), but violated by Python doing the main loop and state management. Fix by migrating that
logic to C++ or releasing GIL.
Principle “One tree per search, threads share it”: Data structures support this, but the code doesn’t
utilize multi-thread sharing yet. Implement parallel simulation on one tree.
Principle “Asynchronous inference with batching”: The scaffolding (inference queue, worker thread)
exists, but actual asynchronous operation is not live. The coordinator still effectively does
synchronous pseudo-inference (sleep 1ms) for each request . This needs to be replaced with
actual batching via GPUInferenceWorker . Once real inference is in, careful tuning of the micro-
batch size and timeout will be needed to hit that 80–90% GPU utilization sweet spot .
Principle “Cache everything sensible”: The code does cache legal move masks in game state and uses
fast C++ for it. Tree data is laid out for cache locality. One possible additional cache: storing neural
network policy for expanded nodes to avoid recomputation if the same state is revisited. However, in
MCTS we rarely revisit exactly the same state in one search, except via transposition (which isn’t
addressed – presumably no transposition table is used, given the complexity for games like Go/
Chess). It’s probably okay not to cache NN results beyond the current tree (AlphaZero doesn’t
traditionally reuse evaluations across different root states except by tree reuse between moves).
In summary, the project’s documentation reflects an excellent conceptual design that the current
code only partially fulfills. The gap is primarily in performance engineering and integration rather than
conceptual flaws. The next development phase should focus on closing that gap: make the
implementation live up to the documented architecture. It’s often wise to prioritize simplifying where
possible: for example, if C++ multi-threading proves too complex to integrate immediately, one might
temporarily pursue a simpler route like using Python’s multiprocessing (separate processes for MCTS
that avoid GIL altogether). That might introduce IPC overhead, but it could be another avenue to parallelism
if GIL remains a problem. Ultimately, though, the intended shared-tree approach is the right long-term
solution for strength and efficiency.
3. Recommendations & Next Steps
Primary Optimizations:
1. Fix MCTS State Transition Bug: This is non-negotiable – correct the misuse of make_move . The
simplest fix is to call apply_move_inplace on the cloned state and drop the redundant clone inside
make_move . For instance:
current_state = current_state.clone()
current_state.apply_move_inplace(move)
instead of the current double-clone logic. Ensure that after this fix, each simulation actually progresses
through a sequence of moves down the tree. This will also allow the tree’s terminal detection to function
properly (so nodes get marked terminal when game ends) and backup to use actual game results . A
quick test after this fix would be to run a few simulations and verify that AlphaZeroMCTS.tree_size
equals the number of simulations (for an empty tree each simulation should add ~1 node until terminal
states or fully expanded nodes start to limit growth).
Migrate the Simulation Loop into C++ (or Cython): Removing Python from the per-simulation path
is the single biggest performance win. Two approaches:
C++ Approach: Write a method, e.g. AlphaZeroMCTS.run_n_simulations(simulations:int) ,
exposed via PyBind11, that executes the for sim in range(simulations):
_run_simulation() loop internally in C++. This method can release the GIL at its start (using
py::call_guard<py::gil_scoped_release>() in the binding) so that Python is not holding it.
Inside, it would call the C++ versions of selection/expansion/backup that are already used. Note that
_run_simulation() itself is currently Python calling a series of C++ and Python steps. You might
need to port some of that logic (particularly the game state interactions) into C++. One idea: push
game state management into C++ as well, by providing the C++ tree with callbacks or function
pointers for expanding and cloning states. This is complex with the current design because
GameState is an interface known only in Python. An intermediate solution is to precompute all
legal moves at the Python level and pass them into C++ (but that breaks the loop of state changes).
Likely, the better route is to use Cython: you could rewrite _run_simulation in a Cython .pyx file,
using the C++ tree and the Python game state calls, but with the GIL released around the loop.
Cython can call Python methods (like GameState.clone() ) with the GIL, but do the looping and
C++ calls without it. This is a more advanced refactor but aligns with the plan mentioned in the guide
. Either way, the goal is to drastically cut the overhead per simulation. By one estimate, if Python
overhead per sim is ~1ms now (as evidenced by the 1k sims/sec when inference is trivial), moving to
C++ could reduce that to ~0.1ms or less for the non-inference parts – a huge difference.
Multi-Thread within this Loop: Once the loop is in C++ and GIL-free, you can add parallelization. For
example, use OpenMP pragmas or std::threads to split the simulation iterations among threads. The
tree’s atomic/virtual loss design should make this thread-safe, but thorough testing is needed. Start
with a fixed number (e.g. 8 threads) and measure scaling. This will directly multiply simulations/sec if
done right (with diminishing returns as threads increase due to contention near the root). The tuning
scripts expected an optimal ~8-10 threads with <10% contention – which seems reasonable.
Achieving that means the virtual loss mechanism is working (to prevent all threads from dogpiling
the same most promising branch). Some adjustments to virtual loss magnitude or locking might be
needed after empirical observation (e.g. if contention is still high, maybe increase virtual loss or
reduce number of threads slightly).
Leverage In-Place State Updates / Avoid Redundant Cloning: As mentioned, using
apply_move_inplace can eliminate one clone per move. We can take this further: consider
reusing a single GameState object for the entire simulation process. A typical pattern in C++ MCTS
is to have one state per thread that is reused for each simulation, doing moves and undos rather
than new copies. Your game state C++ code supports clone() and also an undo_move() . It
should be feasible to implement _run_simulation such that it:
Starts with state = root_state.clone() once at the beginning of search, keep it around.
For each simulation, do a loop from root: at each node, pick move, call
state.apply_move_inplace(move) to advance. If leaf reached, evaluate network, then while
backing up, call state.undo_move() repeatedly to go back up to root.
This way, you allocate at most one clone per search thread (instead of per simulation * per step). The
tricky part is managing this in multi-thread context – each thread would need its own working state
instance (which is fine, 8 threads = 8 states). Also, undo_move must be correct and efficient. The
game C++ code (e.g. Gomoku) would need to maintain move history or stack for undos, which it
appears to have ( get_move_history() exists , and undo_move() is provided ). If those
are reliable, this approach could vastly reduce memory churn and improve cache usage.
Even if implementing full undo is complex, at least aim to minimize clones: currently we might be
cloning O(depth) times per simulation. We can bring that down to 1 clone per simulation (clone root,
then apply moves without further cloning intermediate positions). This change is mostly about
adjusting the _run_simulation logic to not call current_state.clone() at every step, but
only once at the top and then reuse new_state through the selection until a leaf is expanded.
(Since you break out of the loop on expansion, you can then reuse that new_state for backup or
simply discard it after undoing moves. Ensure no stale state persists between simulations.)
Integrate the Real GPU Inference Pipeline: Once the MCTS loop is faster and possibly multi-
threaded, the next bottleneck will be the neural network inference. The code is ready to batch
requests: multiple threads call SearchCoordinator.request_inference() , which places the
game state into a shared inference_request_queue . The GPUInferenceWorker
thread should then pull batches from this queue and evaluate them together on the GPU. Right now
_process_inference_request just fakes this. You should implement
_process_inference_request to actually use self.inference_worker (the GPU worker).
For example, the coordinator could maintain a local batch list, accumulate requests until
batch_size or timeout is reached, then call a method on GPUInferenceWorker to process
that list. In fact, GPUInferenceWorker likely already has an internal loop waiting on its own input
queue (maybe via an InferenceWorker interface defined in specs/inference_api.py ). It may
be easier to bypass the extra queue and call the model directly for each request (less efficient) – but
given all the work put into dynamic batching, you should use it. Test this integration carefully with
various loads: e.g. see that if 8 requests come in nearly simultaneously, they get batched into one
GPU forward pass. Monitor GPU usage; adjust batch_size and timeout_ms as needed. A
default of 64 and 3ms is given , which is a good starting point. The dynamic batching logic in
GPUInferenceWorker will try to adapt the batch size to GPU utilization automatically .
Keep an eye on edge cases: if the self-play generates states slowly, the worker might sit waiting for a
batch and introduce latency – ensure the timeout triggers so it doesn’t wait too long for a full batch.
Also implement the fallback to CPU inference (or at least test that a GPU failure or OOM doesn’t
crash the whole system – the code for fallback is there , but it's complex and might need real-
world testing).
Optimize Move Mapping Storage: As discussed, replace _move_mapping (Python dict) with a
more efficient mechanism. An easy improvement: use a list of length max_nodes (or a numpy
array) to store moves. Python list access by index is O(1) and much lower overhead than dict hashing.
For example, initialize self._moves = [-1] * max_tree_size in
AlphaZeroMCTS.__init__ , and in _expand_node , when adding children, fill in
self._moves[child_index] = move for each new child . Then _get_move_for_child
can simply do move = self._moves[child_index] (no loops or dict lookups) . This will
drastically reduce overhead when the tree grows large, and also simplify memory management (no
need to .clear() the mapping each search; you could just reuse the array and maybe mark a
range of it as used). If memory usage of this array (on the Python side) is a concern (10 million
integers ~ 40 MB, which is not too bad), it could even be allocated in C++ as an additional array in
MCTSTree (e.g. a parallel moves_ array of type uint16 or int32). But Python list/array is fine for now
and much better than dict. After this change, verify that for each expanded node, the move is
correctly recorded (maybe write a quick unit test with a tiny game where you know the moves and
see that _get_move_for_child returns the expected moves).
Secondary Improvements & Clean-up:
- Tree Clearing Optimization: If performance tests show that tree.clear() is taking significant time
(e.g. profiling indicates a lot of time in memset during resets), consider optimizing it. One idea: instead of
fully zeroing arrays, maintain a stack of freed node indices. Since in self-play you usually create a new MCTS
tree for each move anyway, another strategy is tree reuse for subsequent moves. AlphaZero often reuses
the subtree of the chosen move to start the next search (this saves some simulations). Omoknuni’s current
design doesn’t directly support that (no explicit function to set a new root without clearing), but you could
implement a method to re-root the tree at a given child node. Essentially, take the best move’s child node
and treat it as new root (index remapping or swapping arrays). This is non-trivial in an SoA structure
(pointers aren’t used, but you’d have to logically promote a subtree). It may be too much work for now. As a
simpler stopgap, you could reduce max_tree_size from 10 million to something more reasonable
during development (say 1 or 2 million) to make clears faster – but keep in mind final targets might need
the larger size for Go or chess.
Test on Smaller Problems First: Create a simpler environment to test MCTS changes, e.g. a dummy
game with very small action space (like Tic-Tac-Toe or a 4x4 Gomoku). This allows running, say, 10k
simulations on a single thread quickly to measure raw throughput and detect any logic errors in tree
search after your modifications (because you can brute-force check results on a small game). The
existing MockGameState and MockNeuralNetwork in tests provide a starting point for such
controlled testing . Use them with the real AlphaZeroMCTS to ensure everything functions
with the GIL released and multi-threading – you might catch issues like race conditions or incorrectly
applied virtual losses in a deterministic small setting before running on full Gomoku.
Monitoring and Telemetry: Once performance is improved, re-enable or fine-tune telemetry
metrics. For example, tracking simulations_per_second in SearchCoordinator.metrics is
useful, but make sure the way it’s computed is accurate once multi-threading is on (the code
currently increments metrics.total_simulations by request.simulations for each search
and computes throughput periodically – if threads share a tree, you might need to
change how you count completed simulations vs. scheduled). Also verify that the thread utilization
metric (active_threads/ max_threads) still makes sense when some threads might be helping one
search rather than doing separate ones . Minor adjustments might be needed to interpret
metrics in the new scheme.
Error Handling & Stability: With the system becoming more complex (especially multi-thread
interactions and GPU usage), robust error handling becomes important. The code already has
placeholders for that (like catching CriticalInferenceError and triggering emergency
shutdowns). Test these pathways: e.g. simulate a GPU OOM by artificially reducing available memory,
or simulate a model crash, and see if the CPU fallback engages gracefully . Since these are
advanced features, it’s okay if they are not perfect in alpha, but having some confidence that a failed
inference won’t hang a search thread (deadlock) or that a thread crashing won’t silently corrupt the
tree would be good. One possible issue: if a search thread fails mid-way (exception in
_execute_search ), the code currently would leave virtual_loss on some nodes unless
cleaned up. The VirtualLossGuard in _run_simulation should remove it when leaving scope
, but if an exception escapes the entire search, does the code properly reset the tree state? It
might be worth adding a try/finally in AlphaZeroMCTS.search to call reset() on the tree if an
error aborts a search – to avoid contaminated state for the next use. These are edge cases, but in a
training loop that runs indefinitely, even low-probability issues will surface eventually.
Real-World Testing: Finally, after implementing these improvements, run the full self-play training loop in a
controlled setting. Generate some self-play games to see that everything works end-to-end: MCTS suggests
moves, games finish (no infinite loops or crashes), and data is collected. Pay attention to performance
numbers: measure simulations/sec, GPU utilization (perhaps by adding temporary logging or using
NVIDIA’s nvidia-smi dmon tool). If the numbers are still below expectations, identify the next bottleneck.
For example, if after optimizations you reach, say, 8k sims/sec on one thread, but multi-threading only
scales it to 15k on 8 threads (instead of ~88k = 64k), then contention or GIL might still be limiting – you’d
investigate if perhaps the GIL isn’t fully released (maybe due to Python GIL being reacquired each time an
inference future is set or similar). You might then consider further steps like offloading game state logic entirely to
C++* (writing Gomoku move generation and win-checking in C++ to avoid Python calls even for game
methods). In fact, the project already has C++ implementations for Gomoku/Chess/Go logic
( cpp_extensions/games/ ); ensure those are being used via the GameStateWrapper . The
wrapper currently calls alphazero_py C++ methods for moves, so that’s good – but double-check that we
aren’t accidentally using a Python fallback (the code tries alphazero_py and falls back to a Python
adapter if not available ). On a properly installed setup, it should use the C++ game states, which is
crucial for speed.
Conclusion: The Omoknuni project has a strong foundation but requires a concerted optimization effort to
meet its alpha release goals. By fixing logical errors, reducing Python’s role in MCTS, and exploiting full
parallelism, we can expect dramatic improvements. For example, eliminating the double-clone bug and
moving to in-place moves might boost performance from ~1k to ~2-3k sims/sec immediately (less overhead
per sim). Taking the loop out of Python could then raise it into the ~5-10k range single-thread. And enabling
8 threads could feasibly push it into the 30-40k range, matching the target on the given hardware. Each
step must be done carefully to preserve correctness. The reward is substantial: achieving those throughput
figures will not only satisfy deployment requirements but also ensure the engine can generate the volume
of self-play games needed for effective training, with high-quality searches backing each move.
Overall, the next phase is to refine and iterate: profile, optimize one aspect, test, then optimize the next.
This kind of deep performance tuning is challenging but clearly anticipated by the project’s design. With
these recommendations implemented, Omoknuni stands a much better chance of delivering on its promise
of a “High-Performance AlphaZero Engine” , ready for serious alpha/beta testing and ultimately real-world
competitive play and training.
Sources: The analysis above is supported by the project’s code and docs: MCTS engine code , bug
evidence in move application , design guidelines from mcts_guide.md , and test expectations
for performance , among others. These references highlight the discrepancies and guide the solutions
proposed.

-----

Root Cause Analysis: After deep inspection of the codebase, the extremely slow MCTS self-play is primarily due to a mismatch between the intended multi-threaded design and the actual single-threaded implementation. The code as written is not leveraging parallelism or the GPU effectively, causing very low hardware utilization. Key findings:

    No Multi-Threading in MCTS Simulations: The MCTS search is running on a single thread per game, despite the design goal of using 8–12 threads
    GitHub
    GitHub
    . In AlphaZeroMCTS.search(), simulations are executed sequentially in a Python loop (for sim in range(simulations))
    GitHub
    with no internal thread pool or OpenMP parallelism. This means only one CPU core is doing all simulations for a given search, severely capping throughput.

    Python GIL and Loop Overhead: The simulation loop and tree traversal are managed in Python (AlphaZeroMCTS._run_simulation uses a Python while True loop for selection/expansion)
    GitHub
    . Each simulation performs multiple Python->C++ calls (for PUCT selection, node expansion, backup, etc.). Because the GIL is not explicitly released in these calls, multiple threads cannot truly run in parallel here – the GIL will serialize the hot loop. In effect, even if we spawn threads, the heavy computation sections (selection, backup) are holding the GIL, and only one thread executes at a time. This also adds Python function-call overhead in the inner loop of MCTS, slowing each simulation.

    Underutilized GPU Inference Pipeline: The “asynchronous” neural network inference is not actually integrated. The SearchCoordinator currently simulates GPU inference by sleeping 1ms and returning a random policy
    GitHub
    GitHub
    . In other words, the GPU is idle – the real GPUInferenceWorker thread is started but never receives any requests. This leads to near-zero GPU utilization. Furthermore, with only one simulation thread active, even if GPU were used, it would issue inference requests one at a time, never reaching the 32–64 batch size needed for high GPU throughput. The design calls for micro-batching of inference (up to 3ms delay to gather ~32 positions) to reach ~80–90% GPU utilization, but because only 1 or a few requests trickle in, the GPU would mostly wait or run very small batches (inefficient kernel launch overhead).

    Thread Pool Misuse: The SearchCoordinator.max_threads (default 8) is used to allocate a ThreadPoolExecutor, but this pool is used to run separate game searches, not to parallelize a single search. In self-play, if parallel_games is small (e.g. 4), at most 4 threads are active – each running its own game sequentially. The result is low total CPU usage (only a few cores busy). The code currently does not spawn multiple worker threads to collaborate on one MCTS search tree. The intended “shared-tree MCTS” with virtual loss is implemented in theory (atomic counters, VirtualLossManager for thread safety), but it’s not actually being exploited due to the sequential loop.

    Potential Secondary Factors: Each simulation clones the game state and applies a move for every tree step
    GitHub
    , which could be heavy if done in Python. However, game state operations are in C++ and likely efficient (bitboard and in-place move application). The major slowdown is not the game logic, but the lack of parallelism and waiting on inference. Indeed, with ~800 simulations per move and a dummy 1ms inference delay on each new node expansion, a single move search can take on the order of 0.5–1.0 seconds or more, yielding far fewer than the target ~30k simulations/sec. This explains the “horribly slow” self-play (likely tens of games per hour rather than 200+).

Performance Target vs Reality: In summary, the code is falling short of the spec’s performance goals. We expect ~30–40k simulations/sec with ~8 threads and ~80% GPU usage
GitHub
GitHub
. Instead, we have perhaps a few thousand sims/sec on one CPU core, and ~0% GPU. This is definitely not “48-hour training” ready.

Optimization & Debugging Plan:

    Confirm Baseline and Profile: First, verify the current throughput and resource usage to quantify the problem. Use the provided benchmarking tools (e.g. python scripts/test_mcts.py --game gomoku --simulations 1000 --threads 8) and system monitors. We expect to see that increasing --threads above 1 yields almost no speedup, confirming the GIL/serialization issue. Check MetricsCollector outputs or logs for thread_utilization – it will likely show ~12.5% with 1 active thread out of 8
    GitHub
    . Also use nvidia-smi or PyTorch profiler to confirm the GPU is basically idle during self-play. This will solidify that the bottleneck is CPU-bound MCTS and that multi-threading isn’t engaged.

    Enable True Multi-Threaded MCTS: We need to refactor the MCTS search to utilize multiple threads in parallel on a single tree. There are a couple of approaches:

        Python Threading with GIL Release: An immediate fix is to spawn several Python threads in AlphaZeroMCTS.search() to run simulations concurrently. For example, split the simulations count among N worker threads (roughly simulations/threads each) and have each thread call self._run_simulation() in a loop. To make this effective, we must release the GIL inside the C++ extension calls. We will update the pybind11 bindings so that heavy methods release the GIL, e.g. use py::call_guard<py::gil_scoped_release>() for PUCTSelector.select_child, the backup function, and any long-running loops in C++
        GitHub
        GitHub
        . With those changes, the Python threads can truly execute _run_simulation concurrently in C++ without blocking each other. The VirtualLoss mechanism and atomic updates in the C++ MCTS should handle thread conflicts over the shared tree (ensuring threads don’t explore the same leaf without penalty).

        C++ Internal Threads or OpenMP: For maximal performance, a deeper change is to move the simulation loop entirely into C++. We could create a C++ function (in cpp_extensions/mcts) that spawns multiple std::threads or uses OpenMP to run simulations in parallel, calling into the same tree structure. This would bypass Python threading and GIL issues completely. Given the project already uses -fopenmp compile flags
        GitHub
        , we can consider an OpenMP parallel for: e.g. #pragma omp parallel for num_threads(N) over the simulation count. Inside that loop, each iteration would perform selection/expansion/backup for one simulation. We’d need to be careful to call the Python inference_fn from C++ threads – likely by acquiring the GIL for the inference call or using the existing async inference pipeline (see next point). This approach is more complex but could yield near-linear scaling across cores. As a first step, we might implement the Python-thread approach (which is simpler to verify), and then iterate to migrate it into C++ once we see the benefits.

    Validation: After implementing multi-thread search, run benchmarks (e.g. the thread tuning script) to measure improvement. We expect a dramatic jump in simulations/sec and CPU utilization. For example, with 8 threads, CPU usage should approach 800% and sims/sec should multiply several-fold. We’ll also watch out for any race conditions (the atomic counters and VirtualLossGuard should prevent incorrect behavior; we’ll run the thread-safety tests to confirm no crashes or invalid searches).

    Fix the Inference Coordination (Use the GPU): The current dummy inference must be replaced with the real GPU AlphaZeroNet model inference. The architecture is already there: a dedicated GPUInferenceWorker thread with input and output queues for batching. We need to properly route MCTS evaluation requests into that worker:

        Modify SearchCoordinator._process_inference_request() to stop using random results and instead feed the request to the GPUInferenceWorker. Concretely, extract the state’s feature tensor (e.g. via game_state.extract_features() – the code comments indicate this is the intent
        GitHub
        ). Then, enqueue the request to the GPU worker’s input queue. We can leverage the thread_id field in our InferenceRequest to route the result: the GPUInferenceWorker already uses a list of output queues indexed by thread_id
        GitHub
        . For example, we can do:

        self.inference_worker._input_queue.put(request)
        result = self.inference_worker._output_queues[request.thread_id].get(timeout=...)
        request.result_future.set_result((result.policy, result.value))

        Essentially, the SearchCoordinator’s background inference_coordinator_thread becomes a bridge, handing off to the GPU worker and waiting for the result. This preserves the async batching: multiple MCTS threads will place requests, the GPU worker will batch them (waiting up to 3ms or until batch size 32) in its inference_loop
        GitHub
        GitHub
        , and results will come back when ready. We must ensure to handle timeouts or exceptions (if the GPU worker fails or if the queue is full).

        Another approach is to extend the GPUInferenceWorker with a direct async interface. For example, implement a method GPUInferenceWorker.submit(state, thread_id) -> Future that internally creates an InferenceRequest and returns a Future that is completed when the result is ready. This would hide the queue logic inside the worker. Given the current design, integrating via the coordinator’s existing Future mechanism is fine.

    Validation: Once integrated, we should see GPU utilization climb when self-play runs. Use the provided scripts/test_inference.py or simply monitor that the GPU worker’s metrics (batch sizes, latency) are being logged. During MCTS, batches should accumulate (e.g. if we have 8 parallel simulations hitting new leaf nodes, we might see batch sizes in the range 8 or more; with multiple games or threads, possibly up to the 32 limit). The GPU should reach the 80%+ utilization target
    GitHub
    . It’s crucial to verify that the self-play pipeline still produces correct results (the neural net outputs will affect move selection). We’ll run the integration tests and perhaps a small self-play to ensure moves are reasonable (not random anymore).

    Eliminate Python Bottlenecks: With multi-threading and proper GPU usage in place, we should address any remaining Python overhead:

        Move as much of the simulation loop as possible into C++/pybind or Cython. For example, the selection logic in _run_simulation can potentially be handled by a C++ function that returns the final leaf node and value, instead of stepping node by node in Python. This would reduce Python function call overhead and let us use efficient C++ loops. Since the C++ core (PUCT, backup, etc.) is already implemented, this mainly involves shifting the loop control. We might implement a method like MCTSTree.search_batch(simulations, inference_fn) in C++ that manages the parallel loop of simulations – essentially the OpenMP approach described earlier. This is an advanced optimization; we should do it after confirming basic concurrency works, to avoid introducing too many new variables at once.

        Ensure the GIL is released not just for selection but also during any long waits. For instance, waiting on future.result() for inference in each thread will release the GIL internally (since it’s blocking on another thread’s event), which is good – it allows other threads to proceed. We should double-check that pybind functions that might run long (like backup_value_along_path if the path is long) are also wrapped with nogil. According to CLAUDE.md, “All hot loops in C++/Cython with nogil blocks” is a goal
        GitHub
        – we’ll enforce that.

    Memory and Other Tweaks: Once the big issues are fixed, we can consider minor optimizations if needed:

        Transposition Table: The config suggests a transposition table is available (e.g. enable_transposition_table: True in MCTSConfig
        GitHub
        ), but I didn’t see its use in the current code. If not implemented, the search might be redundantly exploring duplicate states (especially in games like Go or Chess). Implementing and using a TT could reduce repeated NN evaluations, boosting speed. However, this is secondary to achieving core throughput.

        State Management: The current approach clones the game state at each step for simulation. In a hot loop, allocating many state copies could be costly. If profiling shows a lot of time spent in GameState.clone(), we might introduce a state-pool or reuse mechanism. Another idea is to avoid Python-level cloning by using a stack of moves: apply moves in C++ and undo them when backing up (this would avoid creating new state objects). These changes are complex and should be approached after addressing the higher-level parallelism issues.

        Batching Inference Requests from One Simulation: In truly asynchronous MCTS, one thread could continue exploring other parts of the tree while a leaf evaluation is pending. Our current design still waits at each expansion (each simulation halts until its NN result comes back). A more advanced optimization would be to allow a thread to suspend that simulation and start another while waiting. This requires a scheduler or coroutine-like approach and is likely too involved. Instead, by having many simulations running in parallel threads, we effectively approximate this – other threads fill the gap while one is waiting for inference. We’ll stick to that model.

    Testing & Iteration: After implementing the above changes, rigorously test the system:

        Functional tests: Run all unit and integration tests (pytest tests/). The self-play outcomes should still be valid (games end correctly, no illegal moves, etc.). Watch for any race conditions introduced by multi-threading (e.g. did we properly protect all shared data? The atomic increments in C++ should cover node stats, but we should test with ThreadSanitizer or run the 1-hour soak test
        GitHub
        to ensure stability).

        Performance tests: Use the performance suite (pytest -m performance) and examine metrics. We expect to meet or come close to the spec: ~30k sims/sec, high GPU utilization, and significantly more games/hour. Particularly, run the thread tuning script again – it should now show an optimum around 8-12 threads with much higher throughput than before. Also run tune_batch_size.py and tune_timeout.py if needed to fine-tune the inference worker settings for the new parallel regime.

        48-hour training dry-run: Ultimately, do a trial run with gomoku_48h_training.yaml for an hour or so. Monitor that it generates self-play games at the expected rate (>=200 games/hour for Gomoku). Check GPU memory usage and ensure no memory leaks or slowdowns over time (the tree reuse and node allocation should keep memory bounded, as per design).

Detailed To-Do Summary:

    Task 1: Release GIL in C++ Hotspots – Modify cpp_extensions/mcts/python_bindings.cpp to add py::gil_scoped_release guards on performance-critical functions (PUCTSelector.select_child, BackupManager.backup_value_along_path, possibly MCTSTree.add_root_node and others that run per simulation in loops). Verify that these functions can indeed run without GIL (they mostly operate on C++ data structures and use atomics, which should be fine).

    Task 2: Implement Parallel Simulation Loop: In AlphaZeroMCTS.search(), introduce multi-threading. E.g., use Python’s threading.Thread or ThreadPoolExecutor to launch num_threads = self.search_coordinator.max_threads worker threads, each executing _run_simulation() in a loop (each should run roughly simulations/num_threads simulations). Use a thread-safe counter or simply divide the work evenly. Make sure to handle any remainder simulations. All threads will share the same AlphaZeroMCTS instance and tree – this is safe now due to the atomic ops and virtual loss. Ensure we do thread.join() before collecting results. (Later, consider replacing this with an all-C++ approach, but start with Python threads for simplicity.)

    Task 3: Refactor Inference Coordination: Overhaul SearchCoordinator.request_inference and _process_inference_request:

        Change _process_inference_request to forward requests to self.inference_worker. Probably define a proper InferenceRequest format for the GPU worker (it may expect a slightly different object). Possibly adapt the existing InferenceRequest dataclass to include the state’s features and thread_id.

        Use the GPU worker’s queues as discussed to get the result. Remove the np.random.dirichlet and dummy sleep. Instead, actually call game_state.get_legal_moves() to mask the NN policy output (the NN will output for all moves, we must zero out illegal ones, which the _expand_node code already does after getting the policy
        GitHub
        ).

        Handle errors: If the GPU fails or times out, ensure we fall back to CPUInferenceWorker or at least return a reasonable policy (the code already had a fallback to uniform policy on exception
        GitHub
        – preserve this logic).

        Test this by running a single search (e.g. call the MCTS search function directly in a small script) and see that it returns non-random, NN-driven policies.

    Task 4: Integrate and Test End-to-End: Run a few self-play games with the new setup. Log the time per move and GPU usage. We should see a big improvement. For example, where before CPU usage was ~1 core and GPU 0%, now all CPU cores should be busy and the GPU at high load during searches. Check that games still complete correctly and that training examples are collected.

    Task 5: Performance Tuning: Use the provided tuning scripts to find the optimal thread count, batch size, etc., in the new regime. For instance, scripts/tune_threads.py will help verify that using, say, 10 threads yields near-max efficiency (contention should be low given our atomic design, but we might find diminishing returns past a certain point). Also, scripts/tune_batch_size.py and tune_timeout.py can adjust the GPU micro-batch parameters if needed (e.g. perhaps a slightly longer timeout than 3ms if threads are not feeding enough, or adjust min_batch_size). The goal is to consistently saturate the GPU without undue delay.

    Task 6: Code Cleanup and Documentation: Update any documentation (specs or comments) if the implementation deviates from what’s described. For example, if we end up implementing the multi-thread loop in C++ instead of Python, update CLAUDE.md and comments to reflect that. Ensure specs/001-goal-create-spec documents are still in sync. Also add any new tests if necessary (for instance, a test to verify that simulations/sec meets a minimum threshold to prevent regressions).

By systematically executing these steps – enabling parallel MCTS, utilizing asynchronous GPU batching as intended, and removing Python/GIL bottlenecks – we should resolve the low throughput issue. We anticipate achieving the targeted 30k+ simulations/sec and 200+ games/hour self-play performance, with high CPU/GPU utilization as per design. This will make the training pipeline viable for 48-hour runs and beyond.