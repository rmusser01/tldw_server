import React from "react"

export interface SourceFolderMembershipOption {
  id: string
  name: string
  depth: number
}

interface SourceFolderMembershipMenuProps {
  sourceTitle: string
  folderOptions: SourceFolderMembershipOption[]
  selectedFolderIds: string[]
  onChange: (folderIds: string[]) => void
}

export const SourceFolderMembershipMenu: React.FC<
  SourceFolderMembershipMenuProps
> = ({ sourceTitle, folderOptions, selectedFolderIds, onChange }) => {
  const [open, setOpen] = React.useState(false)

  if (folderOptions.length === 0) {
    return null
  }

  return (
    <div className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Add ${sourceTitle} to folders`}
        className="rounded border border-border bg-surface px-2 py-1 text-[11px] text-text-muted transition hover:bg-surface2 hover:text-text"
        onClick={() => setOpen((current) => !current)}
      >
        Folders
      </button>

      {open && (
        <div
          role="menu"
          aria-label={`Folders for ${sourceTitle}`}
          className="absolute right-0 z-10 mt-1 min-w-[180px] rounded-md border border-border bg-surface p-2 shadow-lg"
        >
          <div className="space-y-1">
            {folderOptions.map((folder) => {
              const isSelected = selectedFolderIds.includes(folder.id)
              return (
                <label
                  key={folder.id}
                  className="flex items-center gap-2 rounded px-2 py-1 text-xs text-text hover:bg-surface2"
                  style={{ paddingLeft: `${folder.depth * 12 + 8}px` }}
                >
                  <input
                    type="checkbox"
                    role="checkbox"
                    aria-label={`Folder ${folder.name}`}
                    checked={isSelected}
                    onChange={() => {
                      const nextFolderIds = isSelected
                        ? selectedFolderIds.filter((folderId) => folderId !== folder.id)
                        : [...selectedFolderIds, folder.id]
                      onChange(nextFolderIds)
                    }}
                  />
                  <span>{folder.name}</span>
                </label>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
