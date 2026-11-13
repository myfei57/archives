package archives

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"
)

// BudgetLimit identifies which dimension of an [ArchiveBudget] was exceeded.
type BudgetLimit string

const (
	// LimitIdentifyPhysicalBytes bounds the bytes physically pulled from
	// the caller's underlying input stream while Identify() probes the
	// format. Re-reads served from the rewind cache do not consume this
	// meter; re-reads caused by seeking do, because they repeat physical
	// input reads.
	LimitIdentifyPhysicalBytes BudgetLimit = "identify physical input bytes"

	// LimitIdentifyCPUBytes bounds the decompressed output produced while
	// Identify() probes formats through a decompressor. Every repeated
	// probe is billed separately because the CPU decompression work is
	// repeated even when the underlying physical bytes come from the
	// rewind cache.
	LimitIdentifyCPUBytes BudgetLimit = "identify decompression CPU bytes"

	// LimitDecompressedBytes bounds the total output bytes handed to
	// handlers through entry readers for the whole budget session,
	// including every archive layer and every derived (Sub/inner) view.
	LimitDecompressedBytes BudgetLimit = "decompressed output bytes"

	// LimitEntries bounds the total number of archive entries enumerated
	// during the session, across all archive layers and derived views.
	LimitEntries BudgetLimit = "archive entries"

	// LimitEntryBytes bounds the output bytes of any single entry. A
	// declared entry size that already exceeds the limit is rejected
	// before its reader is handed out; when the declared size is unknown
	// (or untrusted), actual emitted bytes are accumulated on every Read.
	LimitEntryBytes BudgetLimit = "single entry bytes"

	// LimitDepth bounds how many layers of nested archives may be opened
	// during the session.
	LimitDepth BudgetLimit = "archive nesting depth"
)

// ErrBudgetExceeded is the sentinel wrapped by every [BudgetExceededError].
// It is deliberately distinct from [NoMatch]: an exhausted budget is a hard
// failure, never an ordinary "format did not match" condition.
var ErrBudgetExceeded = errors.New("archive budget exceeded")

// ErrCodecNotBudgetable is returned when a restricted [ArchiveBudget] meets
// a codec whose internal resource usage cannot be proven bounded here.
var ErrCodecNotBudgetable = errors.New("codec not provably bounded under restricted archive budget")

// BudgetExceededError is returned when an [ArchiveBudget] limit is reached.
// It carries the archive chain (outermost archive first) and the item
// (entry name, archive name, or probe) being processed when the limit was
// hit, so callers can report exactly where the budget ran out.
type BudgetExceededError struct {
	// Limit is the dimension that was exceeded.
	Limit BudgetLimit

	// Max is the configured limit for the dimension (0 if not configured).
	Max int64

	// Used is the metered value at the moment the limit was exceeded.
	Used int64

	// Chain is the archive chain, outermost layer first, as known to the
	// budget at the time of the failure.
	Chain []string

	// Item is the entry, archive, or operation that triggered the charge.
	Item string
}

func (e *BudgetExceededError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %s exceeded (limit=%d used=%d)", ErrBudgetExceeded, e.Limit, e.Max, e.Used)
	if len(e.Chain) > 0 {
		fmt.Fprintf(&b, " chain=%s", strings.Join(e.Chain, "/"))
	}
	if e.Item != "" {
		fmt.Fprintf(&b, " item=%q", e.Item)
	}
	return b.String()
}

// Unwrap exposes the [ErrBudgetExceeded] sentinel for errors.Is.
func (e *BudgetExceededError) Unwrap() error { return ErrBudgetExceeded }

// BudgetUsage is a point-in-time snapshot of an [ArchiveBudget]'s meters.
type BudgetUsage struct {
	IdentifyPhysicalBytes int64
	IdentifyCPUBytes      int64
	DecompressedBytes     int64
	Entries               int64
	Depth                 int
}

// ArchiveBudget is a shared resource-budgeting session for identifying,
// extracting, and traversing (possibly nested) archives.
//
// A single ArchiveBudget value is meant to be shared by the top-level view
// and every view derived from it (ArchiveFS.Sub, DeepFS inner file systems,
// and nested archive views opened through [FileSystem]). Derived views never
// reset or replenish the meters: opening another layer or calling ReadDir
// again cannot grant a fresh allowance, so nested compression cannot bypass
// the limits by spawning new views.
//
// Each Max* field constrains one dimension. The zero value means
// "unlimited" for that dimension, and a nil *ArchiveBudget disables
// budgeting entirely, preserving the behavior of the original APIs.
//
// Metering rules (public contract):
//
//   - Identify physical bytes: only bytes actually pulled from the
//     caller's underlying stream are charged. Reads served from the
//     internal rewind buffer are free; seeker re-reads are charged again
//     because the physical read really repeats.
//   - Identify CPU bytes: decompressed output produced while probing is
//     charged per probe. Repeated probes re-run the decompressor even when
//     their physical input comes from the rewind cache, so the CPU meter
//     may legitimately grow while the physical meter stays flat; the two
//     dimensions are never conflated.
//   - Decompressed output: bytes actually emitted through an entry reader
//     are charged on every Read, against both the session-wide total and
//     the per-entry meter. Reusing a cached directory index or a cached
//     inner view costs nothing; re-opening and re-reading an entry really
//     decompresses it again and is therefore billed again.
//   - Entries: every enumerated archive header (including empty entries
//     and directories) counts once, in every layer, on the shared meter.
//   - Entry bytes: a known declared size larger than the remaining budget
//     is rejected before any reader is handed to the handler; unknown or
//     untrusted declared sizes are enforced by accumulating actual bytes.
//   - Depth: each archive layer opened through [FileSystem] (including
//     DeepFS inner file systems on cache miss) is checked against
//     MaxDepth. Reusing a cached inner view does not open a new layer.
//
// Limits of the guarantee: the meters below bound bytes that flow through
// the readers exposed by this package. They do not, and cannot, bound
// arbitrary internal allocations performed by third-party codec
// implementations (decoder windows, block buffers, solid-block
// accumulation, and similar). Callers that need a hard, auditable boundary
// should set Restricted to true, which refuses codecs and container
// formats whose internal allocation behavior cannot be proven bounded
// here; see [SupportedBudgetCodecs] and [ArchiveBudget.CheckRestricted].
//
// Context cancellation is independent of budgeting: a canceled context
// still surfaces its own context error, never a [BudgetExceededError].
type ArchiveBudget struct {
	// MaxIdentifyBytes limits physical input bytes read during Identify.
	MaxIdentifyBytes int64

	// MaxIdentifyCPUBytes limits decompressed output produced by
	// repeated decompression probes during Identify.
	MaxIdentifyCPUBytes int64

	// MaxDecompressedBytes limits total output bytes of the session.
	MaxDecompressedBytes int64

	// MaxEntries limits the total number of entries enumerated.
	MaxEntries int64

	// MaxEntryBytes limits the output bytes of a single entry.
	MaxEntryBytes int64

	// MaxDepth limits nested archive layers (1 = a single archive).
	MaxDepth int

	// Restricted, if true, refuses codecs/formats that are not on the
	// supported list; see [SupportedBudgetCodecs].
	Restricted bool

	mu      sync.Mutex
	phys    int64
	cpu     int64
	output  int64
	entry   int64
	depth   int
	chain   []string
	lastErr *BudgetExceededError // sticky first failure; meters only grow
}

// budgetContextKey carries an *ArchiveBudget through a context.
type budgetContextKey struct{}

// ContextWithBudget returns a context that carries b. All budget-aware
// operations (Identify, Extract, FileSystem, ArchiveFS, DeepFS) honor a
// budget found in their context, which lets one session propagate through
// arbitrarily many derived views without changing their signatures.
func ContextWithBudget(ctx context.Context, b *ArchiveBudget) context.Context {
	if b == nil {
		return ctx
	}
	return context.WithValue(ctx, budgetContextKey{}, b)
}

// BudgetFromContext returns the budget attached to ctx, or nil if none.
func BudgetFromContext(ctx context.Context) *ArchiveBudget {
	if ctx == nil {
		return nil
	}
	if b, ok := ctx.Value(budgetContextKey{}).(*ArchiveBudget); ok {
		return b
	}
	return nil
}

// layerContextKey carries the nesting level and archive chain of the
// parent view when [FileSystem] opens a derived archive view.
type layerContextKey struct{}

type budgetLayer struct {
	level int
	chain []string
}

// ContextWithBudgetLayer returns a context carrying b along with the
// nesting level and archive chain of the parent archive view. FileSystem
// uses it to enforce [ArchiveBudget.MaxDepth] on derived views and to
// annotate budget errors with the full archive chain. Pass level 0 (and a
// nil chain) for a top-level archive.
func ContextWithBudgetLayer(ctx context.Context, b *ArchiveBudget, level int, chain []string) context.Context {
	ctx = ContextWithBudget(ctx, b)
	return context.WithValue(ctx, layerContextKey{}, budgetLayer{level: level, chain: slices.Clone(chain)})
}

func budgetLayerFromContext(ctx context.Context) budgetLayer {
	if ctx != nil {
		if l, ok := ctx.Value(layerContextKey{}).(budgetLayer); ok {
			return l
		}
	}
	return budgetLayer{}
}

// Usage returns a consistent snapshot of all meters.
func (b *ArchiveBudget) Usage() BudgetUsage {
	if b == nil {
		return BudgetUsage{}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return BudgetUsage{
		IdentifyPhysicalBytes: b.phys,
		IdentifyCPUBytes:      b.cpu,
		DecompressedBytes:     b.output,
		Entries:               b.entry,
		Depth:                 b.depth,
	}
}

// Chain returns a copy of the deepest archive chain recorded so far.
func (b *ArchiveBudget) Chain() []string {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return slices.Clone(b.chain)
}

func (b *ArchiveBudget) exceededLocked(limit BudgetLimit, max, used int64, item string) error {
	err := &BudgetExceededError{
		Limit: limit,
		Max:   max,
		Used:  used,
		Chain: slices.Clone(b.chain),
		Item:  item,
	}
	// remember the first failure; a format's Match that probes a reader
	// and ignores its read error cannot erase a tripped budget
	if b.lastErr == nil {
		b.lastErr = err
	}
	return err
}

// Err returns the first budget error recorded during this session, if any.
// It is a backstop for code paths (such as format Match implementations)
// that consume a metered reader's error internally instead of returning
// it. The returned error remains available for the lifetime of the budget.
func (b *ArchiveBudget) Err() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.lastErr == nil {
		return nil
	}
	cp := *b.lastErr
	return &cp
}

// chargeIdentifyPhysical accounts n bytes physically read from the
// underlying Identify input stream. Buffered re-reads never reach here.
func (b *ArchiveBudget) chargeIdentifyPhysical(n int64) error {
	if b == nil || n <= 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.phys += n
	if b.MaxIdentifyBytes > 0 && b.phys > b.MaxIdentifyBytes {
		return b.exceededLocked(LimitIdentifyPhysicalBytes, b.MaxIdentifyBytes, b.phys, "identify")
	}
	return nil
}

// chargeIdentifyCPU accounts n decompressed bytes produced by an Identify
// probe. Each repeated probe calls this independently.
func (b *ArchiveBudget) chargeIdentifyCPU(n int64) error {
	if b == nil || n <= 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.cpu += n
	if b.MaxIdentifyCPUBytes > 0 && b.cpu > b.MaxIdentifyCPUBytes {
		return b.exceededLocked(LimitIdentifyCPUBytes, b.MaxIdentifyCPUBytes, b.cpu, "identify")
	}
	return nil
}

// countEntry charges one enumerated archive entry against the shared
// entry meter. Empty entries and directories count like any other entry.
func (b *ArchiveBudget) countEntry(item string) error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.entry++
	if b.MaxEntries > 0 && b.entry > b.MaxEntries {
		return b.exceededLocked(LimitEntries, b.MaxEntries, b.entry, item)
	}
	return nil
}

// beforeEntry is called before an entry's output reader is handed to a
// handler. When the declared size is known, it is checked against the
// per-entry limit and the remaining session output budget so that a
// fraudulent or oversized header is rejected before any byte is emitted.
// When the size is unknown (sizeKnown == false), nothing is charged here;
// the entry reader accumulates actual bytes instead.
func (b *ArchiveBudget) beforeEntry(item string, declaredSize int64, sizeKnown bool) error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if !sizeKnown {
		return nil
	}
	if b.MaxEntryBytes > 0 && declaredSize > b.MaxEntryBytes {
		return b.exceededLocked(LimitEntryBytes, b.MaxEntryBytes, declaredSize, item)
	}
	if b.MaxDecompressedBytes > 0 && b.output+declaredSize > b.MaxDecompressedBytes {
		return b.exceededLocked(LimitDecompressedBytes, b.MaxDecompressedBytes, b.output+declaredSize, item)
	}
	return nil
}

// chargeOutput accounts n freshly emitted entry bytes against both the
// per-entry accumulator (owned by the calling reader) and the
// session-wide output meter, atomically.
func (b *ArchiveBudget) chargeOutput(entryBytes *int64, n int64, item string) error {
	if b == nil || n <= 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	*entryBytes += n
	b.output += n
	if b.MaxEntryBytes > 0 && *entryBytes > b.MaxEntryBytes {
		return b.exceededLocked(LimitEntryBytes, b.MaxEntryBytes, *entryBytes, item)
	}
	if b.MaxDecompressedBytes > 0 && b.output > b.MaxDecompressedBytes {
		return b.exceededLocked(LimitDecompressedBytes, b.MaxDecompressedBytes, b.output, item)
	}
	return nil
}

// EnterLayer records that an archive view at the given nesting level was
// opened. Root archives open at level 1; derived inner views open at
// parentLevel+1. The budget only keeps a high-water mark, so reusing a
// cached view never opens a new layer and can never reset the depth. It
// returns [ErrBudgetExceeded] (wrapped in [BudgetExceededError]) when
// level exceeds MaxDepth.
func (b *ArchiveBudget) EnterLayer(level int, chain []string) error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.MaxDepth > 0 && level > b.MaxDepth {
		// report the full chain that was attempted, including the layer
		// that could not be opened, plus the offending item
		err := &BudgetExceededError{
			Limit: LimitDepth,
			Max:   int64(b.MaxDepth),
			Used:  int64(level),
			Chain: slices.Clone(chain),
			Item:  chainLabel(chain),
		}
		if b.lastErr == nil {
			b.lastErr = err
		}
		return err
	}
	if level > b.depth {
		b.depth = level
		b.chain = slices.Clone(chain)
	}
	return nil
}

func chainLabel(chain []string) string {
	if len(chain) == 0 {
		return ""
	}
	return chain[len(chain)-1]
}

// newEntryReader wraps r so that every Read advancing the entry stream is
// metered against the budget's per-entry and session output meters. The
// returned reader owns r and closes it exactly once; it never closes any
// stream owned by the caller (r must be the lent-out entry/decompressor,
// not the caller's input stream).
func (b *ArchiveBudget) newEntryReader(r io.ReadCloser, item string) io.ReadCloser {
	if b == nil {
		return r
	}
	return &budgetReadCloser{ReadCloser: r, b: b, item: item}
}

// budgetReadCloser is the metered reader handed to handlers for entry
// contents. Budget errors are returned from Read alongside the bytes that
// triggered them (the bytes are never silently discarded), and all
// subsequent reads keep failing while the limit remains exceeded.
type budgetReadCloser struct {
	io.ReadCloser
	b          *ArchiveBudget
	item       string
	entryBytes int64
	closeOnce  sync.Once
	closeErr   error
}

func (r *budgetReadCloser) Read(p []byte) (int, error) {
	n, err := r.ReadCloser.Read(p)
	if n > 0 {
		if berr := r.b.chargeOutput(&r.entryBytes, int64(n), r.item); berr != nil {
			return n, berr
		}
	}
	return n, err
}

// Close closes the lent entry/decompressor exactly once. It has no access
// to, and therefore can never close, the caller's underlying input stream.
func (r *budgetReadCloser) Close() error {
	r.closeOnce.Do(func() {
		r.closeErr = r.ReadCloser.Close()
	})
	return r.closeErr
}

// budgetCPUReader meters decompressed output produced by an Identify
// probe; repeated probes wrap separate readers and are billed separately.
// A budget failure is sticky: later reads fail even if the underlying
// decoder could satisfy them, so the failure cannot be masked by a
// ReadFull that happens to fill its whole buffer.
type budgetCPUReader struct {
	io.Reader
	b         *ArchiveBudget
	budgetErr error
}

func (r *budgetCPUReader) Read(p []byte) (int, error) {
	if r.budgetErr != nil {
		return 0, r.budgetErr
	}
	n, err := r.Reader.Read(p)
	if n > 0 {
		if berr := r.b.chargeIdentifyCPU(int64(n)); berr != nil {
			r.budgetErr = berr
			return n, berr
		}
	}
	return n, err
}

// supportedBudgetCodecs lists compression codecs whose readers are pure
// streaming decoders: every decompressed byte becomes observable through
// Read, so an outer metered reader accounts for all emitted output.
//
// Even for these codecs, internal decoder state (window buffers, block
// buffers) is allocated inside the third-party implementation and is not
// bounded by the outer reader; the supported list only states that the
// output stream itself is fully meterable.
var supportedBudgetCodecs = map[string]struct{}{
	".gz":  {}, // github.com/klauspost/compress/gzip, pgzip
	".bz2": {}, // github.com/dsnet/compress/bzip2
	".zst": {}, // github.com/klauspost/compress/zstd
	".xz":  {}, // github.com/mikelolasagasti/xz, github.com/ulikunitz/xz
	".lz4": {}, // github.com/pierrec/lz4/v4
	".lz":  {}, // github.com/sorairolake/lzip-go
	".sz":  {}, // github.com/klauspost/compress/s2 (snappy-framed)
	".zz":  {}, // github.com/klauspost/compress/zlib
	".mz":  {}, // github.com/minio/minlz
	".br":  {}, // github.com/andybalholm/brotli
}

// unsupportedBudgetFormats are container formats whose decoders may
// perform internal allocations (solid blocks, seek indexes, per-volume
// buffers) that an outer LimitReader cannot prove bounded. In restricted
// mode they are refused outright rather than given a false guarantee.
var unsupportedBudgetFormats = map[string]struct{}{
	".rar": {}, // github.com/nwaples/rardecode/v2
	".7z":  {}, // github.com/bodgit/sevenzip
}

// SupportedBudgetCodecs returns the extensions of compression codecs that
// a restricted [ArchiveBudget] will allow, sorted lexicographically.
func SupportedBudgetCodecs() []string {
	out := make([]string, 0, len(supportedBudgetCodecs))
	for ext := range supportedBudgetCodecs {
		out = append(out, ext)
	}
	slices.Sort(out)
	return out
}

// CheckRestricted returns an [ErrCodecNotBudgetable] error when b is in
// restricted mode and name (a format extension such as ".rar") is not on
// the supported-codec list. The returned error records the unprovable
// boundary rather than pretending the outer reader constrains internal
// allocations.
func (b *ArchiveBudget) CheckRestricted(name string) error {
	if b == nil || !b.Restricted {
		return nil
	}
	if _, ok := supportedBudgetCodecs[name]; ok {
		return nil
	}
	reason := "internal decoder allocations are not provably bounded by an outer metered reader"
	if _, ok := unsupportedBudgetFormats[name]; ok {
		reason = "container decoder may allocate solid blocks/index buffers outside the output stream"
	}
	return fmt.Errorf("%w: %s: %s", ErrCodecNotBudgetable, name, reason)
}
