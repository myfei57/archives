package archives

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/klauspost/compress/zip"
)

// Sentinel errors returned by [Zip.ExtractParallel]. Callers should use
// [errors.Is] rather than comparing messages.
var (
	// ErrInvalidParallelOptions is returned before any entry is scheduled when
	// the [ExtractParallelOptions] values are invalid.
	ErrInvalidParallelOptions = errors.New("archives: invalid parallel extraction options")

	// ErrEntrySizeLimit is returned (on the entry's reader and in its report)
	// when an entry decompresses more bytes than MaxEntryBytes allows.
	ErrEntrySizeLimit = errors.New("archives: zip entry decompressed byte limit exceeded")

	// ErrTotalSizeLimit is returned when the aggregate number of decompressed
	// bytes across all entries exceeds MaxDecompressedBytes. It is always a
	// terminal failure: extraction stops scheduling new entries and the error
	// is returned even when ContinueOnError is enabled.
	ErrTotalSizeLimit = errors.New("archives: total decompressed byte limit exceeded")

	// ErrPathConflict is returned by the planning phase when two entries claim
	// the same path, when a path is used as both a directory and a file, or
	// when a non-directory entry has descendant entries. Under
	// [ZipConflictSkip] the conflicting entries are instead reported as
	// skipped with this error wrapped.
	ErrPathConflict = errors.New("archives: conflicting paths in zip entries")
)

// maxLinkTargetSize is the maximum number of bytes read from a symlink
// entry's body when resolving its link target.
const maxLinkTargetSize = 32768

// ZipConflictPolicy determines how [Zip.ExtractParallel] resolves ambiguous
// archive layouts discovered during the planning phase, before any handler is
// invoked.
type ZipConflictPolicy int

const (
	// ZipConflictFail (the zero value) makes the planning phase fail with an
	// [ErrPathConflict] error when duplicate paths or directory/file conflicts
	// are detected. No handler is invoked and nothing is extracted.
	ZipConflictFail ZipConflictPolicy = iota

	// ZipConflictSkip keeps the earliest entry for every conflicting path and
	// marks every later, conflicting ordinal as [ParallelEntrySkipped] with an
	// [ErrPathConflict]-wrapping error in its report. All non-conflicting
	// entries are extracted normally.
	ZipConflictSkip
)

// ExtractParallelOptions configures [Zip.ExtractParallel].
type ExtractParallelOptions struct {
	// MaxWorkers is the hard upper bound on the number of entry handlers that
	// may run concurrently. It also bounds the number of entries that may be
	// queued ahead of processing. If zero, it defaults to [runtime.NumCPU].
	// Negative values produce an [ErrInvalidParallelOptions] error.
	MaxWorkers int

	// MaxOpenReaders is the hard upper bound on the number of entry readers
	// open at any point in time across all workers. A handler blocks (while
	// honoring context cancellation) when the cap is reached. If zero, it
	// defaults to MaxWorkers. Negative values produce an
	// [ErrInvalidParallelOptions] error.
	MaxOpenReaders int

	// MaxDecompressedBytes is the hard upper bound on the total number of
	// bytes decompressed across all entries, including symlink target reads.
	// Accounting is driven by actual Read calls, never by header-declared
	// sizes. Zero means unlimited; negative values produce an
	// [ErrInvalidParallelOptions] error. Exceeding the budget terminates the
	// whole extraction with [ErrTotalSizeLimit], regardless of the
	// ContinueOnError setting.
	MaxDecompressedBytes int64

	// MaxEntryBytes is the hard upper bound on the number of bytes a single
	// entry may decompress, including the body read when resolving a symlink
	// target. Like MaxDecompressedBytes, it is enforced on actual Read calls
	// rather than trusted headers. Zero means unlimited; negative values
	// produce an [ErrInvalidParallelOptions] error. An entry that crosses the
	// limit fails with [ErrEntrySizeLimit].
	MaxEntryBytes int64

	// ConflictPolicy selects how duplicate paths and directory/file conflicts
	// are handled during the planning phase. The zero value is
	// [ZipConflictFail].
	ConflictPolicy ZipConflictPolicy
}

// withDefaults validates the options and fills in zero-value defaults.
func (o ExtractParallelOptions) withDefaults() (ExtractParallelOptions, error) {
	if o.MaxWorkers < 0 || o.MaxOpenReaders < 0 || o.MaxDecompressedBytes < 0 || o.MaxEntryBytes < 0 {
		return o, fmt.Errorf("%w: limits must not be negative", ErrInvalidParallelOptions)
	}
	if o.MaxWorkers == 0 {
		o.MaxWorkers = runtime.NumCPU()
		if o.MaxWorkers < 1 {
			o.MaxWorkers = 1
		}
	}
	if o.MaxOpenReaders == 0 {
		o.MaxOpenReaders = o.MaxWorkers
	}
	return o, nil
}

// ParallelEntryOutcome describes how a single planned entry ended up.
type ParallelEntryOutcome int

const (
	// ParallelEntryCompleted means the handler ran and no handler, read,
	// budget, or close error was recorded for the entry.
	ParallelEntryCompleted ParallelEntryOutcome = iota

	// ParallelEntrySkipped means the entry was never handed to a handler
	// because an ancestor directory returned [fs.SkipDir], or because the
	// entry lost a path conflict under [ZipConflictSkip].
	ParallelEntrySkipped

	// ParallelEntryNotRun means the entry was never started because the walk
	// was stopped by [fs.SkipAll], context cancellation, or a terminal fatal
	// error. Every ordinal still gets a report.
	ParallelEntryNotRun

	// ParallelEntryFailed means the entry was processed but at least one
	// handler, read (including CRC), budget, or close error was recorded.
	ParallelEntryFailed
)

// ParallelEntryReport is the final disposition of one archive entry. The
// Reports slice returned by [Zip.ExtractParallel] is indexed by the entry's
// original ordinal in the ZIP central directory, independent of the order in
// which handlers happened to start or finish.
type ParallelEntryReport struct {
	// Ordinal is the zero-based position of the entry in the ZIP central
	// directory, i.e. its index in zip.Reader.File.
	Ordinal int

	// NameInArchive is the unmodified entry name from the archive header.
	NameInArchive string

	// IsDir reports whether the entry is a directory.
	IsDir bool

	// Symlink reports whether the entry is a symbolic link.
	Symlink bool

	// BytesRead is the number of decompressed bytes actually returned by
	// readers opened for this entry, including any symlink target read. It is
	// measured on real Read calls, so it can differ from the header-declared
	// size.
	BytesRead int64

	// Outcome is the terminal state of the entry.
	Outcome ParallelEntryOutcome

	// Err holds any handler, read (including CRC), budget, or close error
	// recorded for the entry. It is nil for completed, cleanly skipped, and
	// cleanly stopped entries.
	Err error
}

// ExtractParallelResult is the outcome of [Zip.ExtractParallel].
type ExtractParallelResult struct {
	// Reports contains exactly one report per ZIP entry, ordered by the
	// entry's original ordinal. No ordinal is ever missing.
	Reports []ParallelEntryReport
}

// ExtractParallel extracts entries from a seekable ZIP archive using a fixed
// pool of workers. Unlike [Zip.Extract], independent entries are decoded
// concurrently: each call to the handler's FileInfo.Open returns a fresh
// reader backed by its own [io.SectionReader] over the source [io.ReaderAt],
// so workers never share a seek cursor, and every reader the handler leaves
// open is closed once the handler returns (readers closed by the handler
// itself are not closed again).
//
// As with [Zip.Extract], sourceArchive must implement both [io.ReaderAt] and
// [io.Seeker]; the caller retains ownership and ExtractParallel never closes
// it.
//
// Planning happens before anything is scheduled: all central-directory
// headers are read, implicit parent directories are inferred, and duplicate
// paths and directory/file conflicts are resolved according to
// opts.ConflictPolicy. The tree is then walked depth-first with every
// directory entry dispatched before its descendants, so a directory handler
// can return [fs.SkipDir] before any child is queued even when the archive
// itself lists children before their parent directory.
//
// Handler start and completion order is unspecified. The returned
// [ExtractParallelResult.Reports] are nevertheless ordered by original
// ordinal, and every ordinal is reported exactly once.
//
// Returning [fs.SkipAll] from any handler stops new scheduling immediately
// and cancels the context passed to in-flight handlers; [fs.SkipAll] itself
// is not an error. Other handler errors either stop the walk (when
// ContinueOnError is false) or are aggregated into the reports and the
// returned error (when it is true). Budget limit errors are always terminal
// and always surfaced: ContinueOnError can never turn an incomplete,
// budget-truncated extraction into a successful one.
//
// Context cancellation is honored while waiting for a worker or a reader
// slot and between reads. ExtractParallel waits for in-flight handlers to
// return; it cannot forcibly stop a Go function that ignores its context.
// No disk extraction is performed and symbolic links are reported as links
// (their target bodies are read under the same byte budgets) rather than
// followed.
func (z Zip) ExtractParallel(ctx context.Context, sourceArchive io.Reader, opts ExtractParallelOptions, handleFile FileHandler) (ExtractParallelResult, error) {
	var zero ExtractParallelResult

	if handleFile == nil {
		return zero, fmt.Errorf("%w: nil file handler", ErrInvalidParallelOptions)
	}
	opts, err := opts.withDefaults()
	if err != nil {
		return zero, err
	}

	sra, ok := sourceArchive.(seekReaderAt)
	if !ok {
		return zero, fmt.Errorf("input type must be an io.ReaderAt and io.Seeker because of zip format constraints")
	}

	size, err := streamSizeBySeeking(sra)
	if err != nil {
		return zero, fmt.Errorf("determining stream size: %w", err)
	}

	zr, err := zip.NewReader(sra, size)
	if err != nil {
		return zero, err
	}

	nodes, roots, err := z.planParallel(zr, opts.ConflictPolicy)
	if err != nil {
		return zero, err
	}

	runCtx, cancel := context.WithCancel(ctx)
	run := &parallelRun{
		z:       z,
		opts:    opts,
		handler: handleFile,
		ctx:     runCtx,
		cancel:  cancel,
		openSem: make(chan struct{}, opts.MaxOpenReaders),
		jobs:    make(chan *zipPlanNode, opts.MaxWorkers),
		stopCh:  make(chan struct{}),
		board:   newReportBoard(len(nodes)),
	}

	// Propagate cancellation of the caller's context as a terminal stop.
	// ExtractParallel owns runCtx but never the caller's ReaderAt.
	go func() {
		select {
		case <-ctx.Done():
			run.markFatal(ctx.Err())
		case <-run.stopCh:
		}
	}()

	var wg sync.WaitGroup
	for range opts.MaxWorkers {
		wg.Add(1)
		go run.worker(&wg)
	}

	run.walk(roots)
	close(run.jobs)
	wg.Wait()
	cancel()

	// Defensive sweep: every ordinal must end up with exactly one report.
	run.board.fillPending(func(idx int) ParallelEntryReport {
		return run.notRunReport(nodes[idx])
	})

	return ExtractParallelResult{Reports: run.board.reports}, run.aggregateError()
}

// parallelRun carries the mutable, shared state of one ExtractParallel call.
type parallelRun struct {
	z       Zip
	opts    ExtractParallelOptions
	handler FileHandler

	ctx    context.Context
	cancel context.CancelFunc

	// openSem bounds the number of concurrently open entry readers.
	openSem chan struct{}
	// jobs is the bounded work queue; a full queue applies backpressure.
	jobs chan *zipPlanNode

	// totalUsed counts decompressed bytes billed by real Read calls.
	totalUsed atomic.Int64

	stateMu sync.Mutex
	stopped bool
	skipAll bool
	stopErr error
	stopCh  chan struct{}

	board *reportBoard
}

// reportBoard stores exactly one report per ordinal and makes finalization
// idempotent so that workers, the dispatcher, and the final sweep can never
// overwrite each other.
type reportBoard struct {
	mu      sync.Mutex
	reports []ParallelEntryReport
	done    []bool
}

func newReportBoard(n int) *reportBoard {
	return &reportBoard{
		reports: make([]ParallelEntryReport, n),
		done:    make([]bool, n),
	}
}

// finalize records r for idx. It returns false when idx was already
// finalized, in which case the caller's report is discarded.
func (b *reportBoard) finalize(idx int, r ParallelEntryReport) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done[idx] {
		return false
	}
	r.Ordinal = idx
	b.reports[idx] = r
	b.done[idx] = true
	return true
}

func (b *reportBoard) isDone(idx int) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.done[idx]
}

// fillPending finalizes any ordinal that somehow never reached a terminal
// state, using makeReport to build its report.
func (b *reportBoard) fillPending(makeReport func(idx int) ParallelEntryReport) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for idx := range b.reports {
		if b.done[idx] {
			continue
		}
		r := makeReport(idx)
		r.Ordinal = idx
		b.reports[idx] = r
		b.done[idx] = true
	}
}

// zipPlanNode is one node of the planned entry tree. Explicit nodes wrap a
// real ZIP entry; implicit nodes are synthesized for parent directories that
// have no entry of their own.
type zipPlanNode struct {
	entry     *zip.File
	ordinal   int
	key       string // cleaned, slash-separated path without trailing slash
	name      string // raw entry name from the header
	isDir     bool
	isSymlink bool
	implicit  bool

	// conflict is set during planning when the node loses under the
	// configured ZipConflictPolicy.
	conflict       bool
	conflictReason string

	parent   *zipPlanNode
	children []*zipPlanNode

	// barrier is closed once a directory entry reaches a terminal state,
	// letting the dispatcher wait for its SkipDir decision. It is nil for
	// implicit directory nodes, which never have handlers.
	barrier chan struct{}
	// skipChildren is set when a directory handler returns fs.SkipDir.
	skipChildren bool
}

// planParallel reads all central-directory headers, validates path layout
// under the chosen conflict policy, and builds a deterministic tree in which
// every directory precedes its descendants.
func (z Zip) planParallel(zr *zip.Reader, policy ZipConflictPolicy) ([]*zipPlanNode, []*zipPlanNode, error) {
	nodes := make([]*zipPlanNode, len(zr.File))
	for i, f := range zr.File {
		z.decodeText(&f.FileHeader)
		info := f.FileInfo()
		key := path.Clean(strings.TrimSuffix(f.Name, "/"))
		node := &zipPlanNode{
			entry:     f,
			ordinal:   i,
			key:       key,
			name:      f.Name,
			isDir:     info.IsDir(),
			isSymlink: info.Mode()&os.ModeSymlink != 0,
		}
		if node.isDir {
			node.barrier = make(chan struct{})
		}
		nodes[i] = node
	}

	// First pass: same-key duplicates and directory/file type clashes.
	byKey := make(map[string]*zipPlanNode, len(nodes))
	var conflictMessages []string
	markConflict := func(node *zipPlanNode, format string, args ...any) {
		if node.conflict {
			return
		}
		node.conflict = true
		node.conflictReason = fmt.Sprintf(format, args...)
		conflictMessages = append(conflictMessages,
			fmt.Sprintf("entry %d %q: %s", node.ordinal, node.name, node.conflictReason))
	}
	for _, node := range nodes {
		prev, exists := byKey[node.key]
		if !exists {
			byKey[node.key] = node
			continue
		}
		if prev.isDir != node.isDir {
			markConflict(node, "path %q is already used by a %s at entry %d",
				node.key, entryKind(prev), prev.ordinal)
		} else {
			markConflict(node, "duplicate path %q first seen at entry %d",
				node.key, prev.ordinal)
		}
	}

	// Second pass: a non-directory entry cannot have descendants.
	for _, node := range nodes {
		dir := path.Dir(node.key)
		for dir != "." && dir != "/" && dir != "" {
			if ancestor, ok := byKey[dir]; ok && !ancestor.isDir {
				later := ancestor
				if node.ordinal > ancestor.ordinal {
					later = node
				}
				markConflict(later, "non-directory entry %q cannot contain descendant %q",
					ancestor.key, node.key)
			}
			parent := path.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	if policy == ZipConflictFail && len(conflictMessages) > 0 {
		return nil, nil, fmt.Errorf("%w:\n\t%s",
			ErrPathConflict, strings.Join(conflictMessages, "\n\t"))
	}

	// Build the tree, synthesizing implicit directory nodes as needed.
	var roots []*zipPlanNode
	var ensureParent func(node *zipPlanNode) *zipPlanNode
	ensureParent = func(node *zipPlanNode) *zipPlanNode {
		parentKey := path.Dir(node.key)
		if parentKey == "." || parentKey == "" || parentKey == node.key {
			return nil
		}
		parent := byKey[parentKey]
		if parent == nil {
			parent = &zipPlanNode{
			key:      parentKey,
			name:     parentKey + "/",
			isDir:    true,
			implicit: true,
			ordinal:  -1,
		}
			byKey[parentKey] = parent
			if grand := ensureParent(parent); grand != nil {
				grand.children = append(grand.children, parent)
			} else {
				roots = append(roots, parent)
			}
		}
		parent.children = append(parent.children, node)
		node.parent = parent
		return parent
	}
	for _, node := range nodes {
		if ensureParent(node) == nil {
			roots = append(roots, node)
		}
	}

	var sortNodes func(ns []*zipPlanNode)
	sortNodes = func(ns []*zipPlanNode) {
		sort.Slice(ns, func(i, j int) bool { return ns[i].key < ns[j].key })
		for _, node := range ns {
			sortNodes(node.children)
		}
	}
	sortNodes(roots)

	return nodes, roots, nil
}

func entryKind(node *zipPlanNode) string {
	if node.isDir {
		return "directory"
	}
	return "file"
}

// worker drains the job queue until it is closed.
func (run *parallelRun) worker(wg *sync.WaitGroup) {
	defer wg.Done()
	for node := range run.jobs {
		if run.board.isDone(node.ordinal) {
			continue
		}
		run.execute(node)
	}
}

// walk dispatches the tree depth-first. A directory is fully resolved
// (handler outcome observed via its barrier) before any descendant is
// enqueued, which gives SkipDir its required ordering guarantee.
func (run *parallelRun) walk(nodes []*zipPlanNode) {
	for _, node := range nodes {
		if run.isStopping() {
			run.sweepNotRun([]*zipPlanNode{node})
			continue
		}
		if node.implicit {
			run.walk(node.children)
			continue
		}
		if node.conflict {
			run.board.finalize(node.ordinal, ParallelEntryReport{
				NameInArchive: node.name,
				IsDir:         node.isDir,
				Symlink:       node.isSymlink,
				Outcome:       ParallelEntrySkipped,
				Err:           fmt.Errorf("%w: %s", ErrPathConflict, node.conflictReason),
			})
			run.walk(node.children)
			continue
		}
		if !run.enqueue(node) {
			run.sweepNotRun([]*zipPlanNode{node})
			continue
		}
		if node.isDir {
			<-node.barrier
			if node.skipChildren {
				run.sweepSkipped(node.children)
				continue
			}
		}
		run.walk(node.children)
	}
}

// enqueue adds a node to the work queue, unblocking when the walk terminates.
func (run *parallelRun) enqueue(node *zipPlanNode) bool {
	select {
	case run.jobs <- node:
		return true
	case <-run.stopCh:
		return false
	}
}

// sweepNotRun finalizes an unprocessed subtree as not-run.
func (run *parallelRun) sweepNotRun(nodes []*zipPlanNode) {
	for _, node := range nodes {
		if !node.implicit {
			run.board.finalize(node.ordinal, run.notRunReport(node))
		}
		run.sweepNotRun(node.children)
	}
}

// sweepSkipped finalizes an unprocessed subtree as skipped by SkipDir.
func (run *parallelRun) sweepSkipped(nodes []*zipPlanNode) {
	for _, node := range nodes {
		if !node.implicit {
			run.board.finalize(node.ordinal, ParallelEntryReport{
				NameInArchive: node.name,
				IsDir:         node.isDir,
				Symlink:       node.isSymlink,
				Outcome:       ParallelEntrySkipped,
			})
		}
		run.sweepSkipped(node.children)
	}
}

func (run *parallelRun) notRunReport(node *zipPlanNode) ParallelEntryReport {
	report := ParallelEntryReport{
		NameInArchive: node.name,
		IsDir:         node.isDir,
		Symlink:       node.isSymlink,
		Outcome:       ParallelEntryNotRun,
	}
	run.stateMu.Lock()
	skipAll, stopErr := run.skipAll, run.stopErr
	run.stateMu.Unlock()
	if !skipAll && stopErr != nil {
		report.Err = fmt.Errorf("entry not processed: %w", stopErr)
	}
	return report
}

// execute runs one entry end to end. It is the single place where an
// enqueued entry reaches its terminal state.
func (run *parallelRun) execute(node *zipPlanNode) {
	if node.isDir {
		// Unblock the dispatcher regardless of how this entry ends.
		defer close(node.barrier)
	}

	select {
	case <-run.stopCh:
		run.board.finalize(node.ordinal, run.notRunReport(node))
		return
	default:
	}

	handle := &entryReaderHandle{run: run, node: node}

	info := node.entry.FileInfo()
	file := FileInfo{
		FileInfo:      info,
		Header:        node.entry.FileHeader,
		NameInArchive: node.name,
		LinkTarget:    "",
		Open: func() (fs.File, error) {
			return handle.open()
		},
	}

	var linkErr error
	if node.isSymlink {
		linkErr = run.populateLinkTarget(handle, &file)
	}

	var handlerErr error
	if linkErr == nil {
		handlerErr = run.invokeHandler(file)
	} else {
		handlerErr = linkErr
	}

	// Always close anything the handler left open, even after an error or
	// panic, and capture close errors instead of dropping them.
	closeErr := handle.closeAll()

	run.complete(node, handle, handlerErr, closeErr)
}

// invokeHandler runs the user handler, converting a panic into a terminal
// error so a panicking handler cannot tear down the worker pool silently.
func (run *parallelRun) invokeHandler(file FileInfo) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("file handler panicked: %v\n%s", recovered, debug.Stack())
			run.markFatal(err)
		}
	}()
	return run.handler(run.ctx, file)
}

// populateLinkTarget reads the symlink body through the same metered,
// budgeted reader gate as ordinary content. The reader is released before
// the user handler runs; links are never followed or extracted to disk.
func (run *parallelRun) populateLinkTarget(handle *entryReaderHandle, file *FileInfo) error {
	linkFile, err := handle.open()
	if err != nil {
		return fmt.Errorf("opening symlink %q: %w", file.NameInArchive, err)
	}
	defer linkFile.Close()

	target, err := io.ReadAll(io.LimitReader(linkFile, maxLinkTargetSize+1))
	if err == nil {
		// A budget breach can arrive on the same read that returns EOF;
		// check the sticky error explicitly before accepting the target so
		// the user handler never runs against a truncated link body.
		err = handle.readSticky()
	}
	if err != nil {
		return fmt.Errorf("reading symlink target %q: %w", file.NameInArchive, err)
	}
	if len(target) > maxLinkTargetSize {
		return fmt.Errorf("symlink target for %q exceeds %d bytes", file.NameInArchive, maxLinkTargetSize)
	}
	file.LinkTarget = string(target)
	return nil
}

// complete interprets the handler outcome together with sticky read errors
// and close errors and finalizes the entry's report.
func (run *parallelRun) complete(node *zipPlanNode, handle *entryReaderHandle, handlerErr, closeErr error) {
	report := ParallelEntryReport{
		NameInArchive: node.name,
		IsDir:         node.isDir,
		Symlink:       node.isSymlink,
		BytesRead:     handle.bytesRead(),
	}

	switch {
	case errors.Is(handlerErr, fs.SkipAll):
		// The entry itself ran; SkipAll is a protocol signal, not a failure.
		run.markSkipAll()
	case errors.Is(handlerErr, fs.SkipDir) && node.isDir:
		node.skipChildren = true
	case handlerErr != nil:
		// During SkipAll, in-flight handlers that surface the cooperative
		// context cancellation are winding down, not failing the walk.
		if errors.Is(handlerErr, context.Canceled) && run.isSkipAll() {
			handlerErr = nil
		} else {
			report.Err = errors.Join(report.Err, handlerErr)
			if !run.z.ContinueOnError {
				run.markFatal(handlerErr)
			}
		}
	}

	// Sticky read errors cover CRC mismatches and budget breaches; surface
	// them even when the user handler swallowed the Read error.
	if sticky := handle.readSticky(); sticky != nil {
		report.Err = errors.Join(report.Err, sticky)
		if errors.Is(sticky, ErrEntrySizeLimit) && !run.z.ContinueOnError {
			run.markFatal(sticky)
		}
	}

	if closeErr != nil {
		report.Err = errors.Join(report.Err, fmt.Errorf("closing reader for %q: %w", node.name, closeErr))
		if !run.z.ContinueOnError {
			run.markFatal(closeErr)
		}
	}

	if report.Err != nil {
		report.Outcome = ParallelEntryFailed
	} else {
		report.Outcome = ParallelEntryCompleted
	}
	run.board.finalize(node.ordinal, report)
}

// aggregateError joins the terminal stop reason (SkipAll excepted) with every
// failed entry's error so that no error is lost while errors.Is keeps working.
func (run *parallelRun) aggregateError() error {
	var errs []error

	run.stateMu.Lock()
	stopErr, skipAll := run.stopErr, run.skipAll
	run.stateMu.Unlock()

	if stopErr != nil && !skipAll {
		errs = append(errs, stopErr)
	}
	for _, report := range run.board.reports {
		if report.Outcome != ParallelEntryFailed || report.Err == nil {
			continue
		}
		// The stop error is already included once.
		if stopErr != nil && errors.Is(report.Err, stopErr) {
			continue
		}
		errs = append(errs, report.Err)
	}
	return errors.Join(errs...)
}

func (run *parallelRun) isStopping() bool {
	select {
	case <-run.stopCh:
		return true
	default:
		return false
	}
}

func (run *parallelRun) isSkipAll() bool {
	run.stateMu.Lock()
	defer run.stateMu.Unlock()
	return run.skipAll
}

// markFatal terminates the walk with a real error.
func (run *parallelRun) markFatal(err error) {
	if err == nil {
		return
	}
	run.stateMu.Lock()
	defer run.stateMu.Unlock()
	if run.stopped {
		return
	}
	run.stopped = true
	run.stopErr = err
	close(run.stopCh)
	run.cancel()
}

// markSkipAll terminates the walk without an error.
func (run *parallelRun) markSkipAll() {
	run.stateMu.Lock()
	defer run.stateMu.Unlock()
	if run.stopped {
		return
	}
	run.stopped = true
	run.skipAll = true
	close(run.stopCh)
	run.cancel()
}

// entryReaderHandle owns every reader session opened for one entry and the
// byte accounting shared by those sessions.
type entryReaderHandle struct {
	run  *parallelRun
	node *zipPlanNode

	mu       sync.Mutex
	sessions []*readerSession
	bytes    int64
	sticky   error // first sticky read error (CRC, budget, ...)
}

// open acquires a reader slot and returns a brand-new, entry-private reader.
// Each call creates an independent [io.SectionReader], so sessions and
// workers never share a seek cursor.
func (h *entryReaderHandle) open() (fs.File, error) {
	select {
	case h.run.openSem <- struct{}{}:
	case <-h.run.ctx.Done():
		return nil, h.run.ctx.Err()
	}

	readCloser, err := h.node.entry.Open()
	if err != nil {
		<-h.run.openSem
		return nil, err
	}

	session := &readerSession{
		fileInArchive: fileInArchive{
			ReadCloser: &meteredReader{rc: readCloser, handle: h},
			info:       h.node.entry.FileInfo(),
		},
		handle: h,
	}

	h.mu.Lock()
	h.sessions = append(h.sessions, session)
	h.mu.Unlock()

	return session, nil
}

// removeSession drops a closed session and releases its reader slot.
func (h *entryReaderHandle) removeSession(session *readerSession) {
	h.mu.Lock()
	for i, candidate := range h.sessions {
		if candidate == session {
			h.sessions = append(h.sessions[:i], h.sessions[i+1:]...)
			break
		}
	}
	h.mu.Unlock()
	<-h.run.openSem
}

// closeAll closes every session the handler left behind. It is safe when the
// handler closed some or all sessions itself.
func (h *entryReaderHandle) closeAll() error {
	h.mu.Lock()
	sessions := append([]*readerSession(nil), h.sessions...)
	h.mu.Unlock()

	var err error
	for _, session := range sessions {
		err = errors.Join(err, session.Close())
	}
	return err
}

func (h *entryReaderHandle) bytesRead() int64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.bytes
}

func (h *entryReaderHandle) readSticky() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.sticky
}

// setSticky records the first read error for the entry so that CRC and
// budget failures cannot be lost when a handler ignores Read errors.
func (h *entryReaderHandle) setSticky(err error) {
	if err == nil {
		return
	}
	h.mu.Lock()
	if h.sticky == nil {
		h.sticky = err
	}
	h.mu.Unlock()
}

// charge bills n decompressed bytes against both the per-entry and the
// global budgets. Breaching either limit installs a sticky error; the global
// breach additionally terminates the whole extraction.
func (h *entryReaderHandle) charge(n int) {
	var entryErr error

	h.mu.Lock()
	h.bytes += int64(n)
	if limit := h.run.opts.MaxEntryBytes; limit > 0 && h.bytes > limit &&
		!errors.Is(h.sticky, ErrEntrySizeLimit) {
		entryErr = fmt.Errorf("%w: entry %q decompressed %d bytes (limit %d)",
			ErrEntrySizeLimit, h.node.name, h.bytes, limit)
	}
	h.mu.Unlock()

	if entryErr != nil {
		h.setSticky(entryErr)
	}

	if limit := h.run.opts.MaxDecompressedBytes; limit > 0 {
		total := h.run.totalUsed.Add(int64(n))
		if total > limit {
			totalErr := fmt.Errorf("%w: %d decompressed bytes in total (limit %d)",
				ErrTotalSizeLimit, total, limit)
			h.setSticky(totalErr)
			h.run.markFatal(totalErr)
		}
	}
}

// readerSession is one independently opened reader for an entry. Closing it
// is idempotent and always releases exactly one reader slot.
type readerSession struct {
	fileInArchive
	handle *entryReaderHandle

	closeOnce sync.Once
	closeErr  error
}

func (s *readerSession) Close() error {
	s.closeOnce.Do(func() {
		s.closeErr = s.fileInArchive.Close()
		s.handle.removeSession(s)
	})
	return s.closeErr
}

// meteredReader wraps the zip library's decompressing reader and bills every
// byte actually returned by Read against the entry and global budgets.
type meteredReader struct {
	rc     io.ReadCloser
	handle *entryReaderHandle
}

func (m *meteredReader) Read(p []byte) (int, error) {
	if sticky := m.handle.readSticky(); sticky != nil {
		return 0, sticky
	}
	if err := m.handle.run.ctx.Err(); err != nil {
		return 0, err
	}

	n, err := m.rc.Read(p)
	if n > 0 {
		m.handle.charge(n)
	}
	if err != nil {
		if err != io.EOF {
			m.handle.setSticky(err)
		}
		return n, err
	}
	if sticky := m.handle.readSticky(); sticky != nil {
		// A budget breach may have been installed by charge(); report it on
		// this read rather than waiting for the next one.
		return n, sticky
	}
	return n, nil
}

func (m *meteredReader) Close() error {
	return m.rc.Close()
}

// Interface guard: ExtractParallel is an additional capability of Zip, not a
// replacement for the sequential Extractor contract.
var _ interface {
	ExtractParallel(context.Context, io.Reader, ExtractParallelOptions, FileHandler) (ExtractParallelResult, error)
} = Zip{}
