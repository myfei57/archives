//go:build !windows

package archives

import "io/fs"

// canonicalInodeKey identifies a physical file for hard-link grouping.
type canonicalInodeKey struct {
	dev uint64
	ino uint64
}

// defaultHardlinkKey adapts the platform inode detection used by
// FilesFromDisk to the platform-neutral canonical archive key type.
func defaultHardlinkKey(fi fs.FileInfo) (canonicalInodeKey, bool) {
	k, ok := hardlinkKey(fi)
	if !ok {
		return canonicalInodeKey{}, false
	}
	return canonicalInodeKey{dev: k.dev, ino: k.ino}, true
}
