import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Button, Empty, Input, List, Segmented, Select, Spin, Tag, Typography, message } from "antd"
import { Plus, Users, Globe, GitBranch } from "lucide-react"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import {
  listManuscriptCharacters,
  createManuscriptCharacter,
  listManuscriptWorldInfo,
  createManuscriptWorldInfo,
  listManuscriptPlotLines,
  createManuscriptPlotLine,
  listManuscriptPlotHoles,
  type ManuscriptCharacter,
  type ManuscriptCharacterListResponse,
  type ManuscriptPlotHole,
  type ManuscriptPlotHoleListResponse,
  type ManuscriptPlotLine,
  type ManuscriptPlotLineListResponse,
  type ManuscriptWorldInfoItem,
  type ManuscriptWorldInfoListResponse,
} from "@/services/writing-playground"

type CharacterWorldTabProps = { isOnline: boolean }
type SubView = "characters" | "world" | "plot"

const ROLE_COLORS: Record<string, string> = {
  protagonist: "blue",
  antagonist: "red",
  supporting: "green",
  minor: "default",
  mentioned: "default",
}

const KIND_LABELS: Record<string, string> = {
  location: "Location",
  item: "Item",
  faction: "Faction",
  concept: "Concept",
  event: "Event",
  custom: "Custom",
}

export function CharacterWorldTab({ isOnline }: CharacterWorldTabProps) {
  const { activeProjectId } = useWritingPlaygroundStore()
  const queryClient = useQueryClient()
  const [subView, setSubView] = useState<SubView>("characters")
  const [newCharName, setNewCharName] = useState("")
  const [newWorldName, setNewWorldName] = useState("")
  const [newWorldKind, setNewWorldKind] = useState<string>("location")
  const [newPlotTitle, setNewPlotTitle] = useState("")

  // ── Queries ──
  const { data: charactersResponse, isLoading: charsLoading, isError: charsError } = useQuery<ManuscriptCharacterListResponse>({
    queryKey: ["manuscript-characters", activeProjectId],
    queryFn: () => listManuscriptCharacters(activeProjectId!),
    enabled: isOnline && !!activeProjectId && subView === "characters",
    staleTime: 30_000,
  })

  const { data: worldInfoResponse, isLoading: worldLoading, isError: worldError } = useQuery<ManuscriptWorldInfoListResponse>({
    queryKey: ["manuscript-world-info", activeProjectId],
    queryFn: () => listManuscriptWorldInfo(activeProjectId!),
    enabled: isOnline && !!activeProjectId && subView === "world",
    staleTime: 30_000,
  })

  const { data: plotLinesResponse, isLoading: plotLoading, isError: plotError } = useQuery<ManuscriptPlotLineListResponse>({
    queryKey: ["manuscript-plot-lines", activeProjectId],
    queryFn: () => listManuscriptPlotLines(activeProjectId!),
    enabled: isOnline && !!activeProjectId && subView === "plot",
    staleTime: 30_000,
  })

  const { data: plotHolesResponse, isError: plotHolesError } = useQuery<ManuscriptPlotHoleListResponse>({
    queryKey: ["manuscript-plot-holes", activeProjectId],
    queryFn: () => listManuscriptPlotHoles(activeProjectId!),
    enabled: isOnline && !!activeProjectId && subView === "plot",
    staleTime: 30_000,
  })

  const characters = charactersResponse?.characters ?? []
  const worldInfo = worldInfoResponse?.items ?? []
  const plotLines = plotLinesResponse?.plot_lines ?? []
  const plotHoles = plotHolesResponse?.plot_holes ?? []

  // ── Mutations ──
  const addCharMutation = useMutation({
    mutationFn: (name: string) => createManuscriptCharacter(activeProjectId!, { name }),
    onSuccess: (_data, name) => {
      queryClient.invalidateQueries({ queryKey: ["manuscript-characters", activeProjectId] })
      setNewCharName((current) => current === name ? "" : current)
    },
    onError: (err: Error) => {
      console.debug("[CharacterWorldTab] Failed to create character", err)
      message.error(err.message || "Failed to create character")
    },
  })

  const addWorldMutation = useMutation({
    mutationFn: ({ name, kind }: { name: string; kind: string }) =>
      createManuscriptWorldInfo(activeProjectId!, { name, kind }),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["manuscript-world-info", activeProjectId] })
      setNewWorldName((current) => current === variables.name ? "" : current)
    },
    onError: (err: Error) => {
      console.debug("[CharacterWorldTab] Failed to create world info", err)
      message.error(err.message || "Failed to create world info")
    },
  })

  const addPlotMutation = useMutation({
    mutationFn: (title: string) => createManuscriptPlotLine(activeProjectId!, { title }),
    onSuccess: (_data, title) => {
      queryClient.invalidateQueries({ queryKey: ["manuscript-plot-lines", activeProjectId] })
      setNewPlotTitle((current) => current === title ? "" : current)
    },
    onError: (err: Error) => {
      console.debug("[CharacterWorldTab] Failed to create plot line", err)
      message.error(err.message || "Failed to create plot line")
    },
  })

  if (!activeProjectId) {
    return <Empty description="Select a manuscript project first" className="py-8" />
  }

  return (
    <div className="flex flex-col gap-3">
      <Segmented
        block
        size="small"
        value={subView}
        onChange={(v) => setSubView(v as SubView)}
        options={[
          { value: "characters", label: "Characters", icon: <Users className="h-3 w-3" /> },
          { value: "world", label: "World", icon: <Globe className="h-3 w-3" /> },
          { value: "plot", label: "Plot", icon: <GitBranch className="h-3 w-3" /> },
        ]}
      />

      {subView === "characters" && (
        <div className="flex flex-col gap-2">
          <div className="flex gap-1">
            <Input
              size="small"
              placeholder="Character name..."
              aria-label="New character name"
              value={newCharName}
              onChange={(e) => setNewCharName(e.target.value)}
              onPressEnter={() => isOnline && newCharName.trim() && !addCharMutation.isPending && addCharMutation.mutate(newCharName.trim())}
            />
            <Button
              size="small"
              type="primary"
              icon={<Plus className="h-3 w-3" />}
              aria-label="Add character"
              disabled={!newCharName.trim() || !isOnline}
              loading={addCharMutation.isPending}
              onClick={() => isOnline && addCharMutation.mutate(newCharName.trim())}
            />
          </div>
          {charsLoading ? <Spin size="small" /> : charsError ? (
            <Typography.Text type="danger" className="text-xs">Failed to load characters</Typography.Text>
          ) : (
            <List
              size="small"
              dataSource={characters}
              locale={{ emptyText: "No characters yet" }}
              renderItem={(char: ManuscriptCharacter) => (
                <List.Item className="!px-0 !py-1">
                  <div className="flex items-center gap-2 w-full">
                    <Typography.Text className="text-sm flex-1">{char.name}</Typography.Text>
                    <Tag color={ROLE_COLORS[char.role] || "default"} className="!text-xs">
                      {char.role}
                    </Tag>
                  </div>
                </List.Item>
              )}
            />
          )}
        </div>
      )}

      {subView === "world" && (
        <div className="flex flex-col gap-2">
          <div className="flex gap-1">
            <Select
              size="small"
              value={newWorldKind}
              onChange={setNewWorldKind}
              options={Object.entries(KIND_LABELS).map(([k, v]) => ({ value: k, label: v }))}
              aria-label="World info kind"
              className="!w-24"
            />
            <Input
              size="small"
              placeholder="Entry name..."
              aria-label="New world info entry name"
              value={newWorldName}
              onChange={(e) => setNewWorldName(e.target.value)}
              onPressEnter={() => isOnline && newWorldName.trim() && !addWorldMutation.isPending && addWorldMutation.mutate({ name: newWorldName.trim(), kind: newWorldKind })}
              className="flex-1"
            />
            <Button
              size="small"
              type="primary"
              icon={<Plus className="h-3 w-3" />}
              aria-label="Add world info entry"
              disabled={!newWorldName.trim() || !isOnline}
              loading={addWorldMutation.isPending}
              onClick={() => isOnline && addWorldMutation.mutate({ name: newWorldName.trim(), kind: newWorldKind })}
            />
          </div>
          {worldLoading ? <Spin size="small" /> : worldError ? (
            <Typography.Text type="danger" className="text-xs">Failed to load world info</Typography.Text>
          ) : (
            <List
              size="small"
              dataSource={worldInfo}
              locale={{ emptyText: "No world info yet" }}
              renderItem={(wi: ManuscriptWorldInfoItem) => (
                <List.Item className="!px-0 !py-1">
                  <div className="flex items-center gap-2 w-full">
                    <Typography.Text className="text-sm flex-1">{wi.name}</Typography.Text>
                    <Tag className="!text-xs">{KIND_LABELS[wi.kind] || wi.kind}</Tag>
                  </div>
                </List.Item>
              )}
            />
          )}
        </div>
      )}

      {subView === "plot" && (
        <div className="flex flex-col gap-3">
          <div>
            <Typography.Text strong className="text-xs">Plot Lines</Typography.Text>
            <div className="flex gap-1 mt-1">
              <Input
                size="small"
                placeholder="Plot line title..."
                aria-label="New plot line title"
                value={newPlotTitle}
                onChange={(e) => setNewPlotTitle(e.target.value)}
                onPressEnter={() => isOnline && newPlotTitle.trim() && !addPlotMutation.isPending && addPlotMutation.mutate(newPlotTitle.trim())}
              />
              <Button
                size="small"
                type="primary"
                icon={<Plus className="h-3 w-3" />}
                aria-label="Add plot line"
                disabled={!newPlotTitle.trim() || !isOnline}
                loading={addPlotMutation.isPending}
                onClick={() => isOnline && addPlotMutation.mutate(newPlotTitle.trim())}
              />
            </div>
            {plotLoading ? <Spin size="small" /> : plotError ? (
              <Typography.Text type="danger" className="text-xs">Failed to load plot lines</Typography.Text>
            ) : (
              <List
                size="small"
                dataSource={plotLines}
                locale={{ emptyText: "No plot lines yet" }}
                renderItem={(pl: ManuscriptPlotLine) => (
                  <List.Item className="!px-0 !py-1">
                    <div className="flex items-center gap-2 w-full">
                      <Typography.Text className="text-sm flex-1">{pl.title}</Typography.Text>
                      <Tag color={pl.status === "resolved" ? "green" : pl.status === "abandoned" ? "default" : "blue"} className="!text-xs">
                        {pl.status}
                      </Tag>
                    </div>
                  </List.Item>
                )}
              />
            )}
          </div>
          {plotHolesError ? (
            <Typography.Text type="danger" className="text-xs">Failed to load plot holes</Typography.Text>
          ) : plotHoles.length > 0 ? (
            <div>
              <Typography.Text strong className="text-xs">Plot Holes</Typography.Text>
              <List
                size="small"
                dataSource={plotHoles}
                renderItem={(ph: ManuscriptPlotHole) => (
                  <List.Item className="!px-0 !py-1">
                    <div className="flex items-center gap-2 w-full">
                      <Typography.Text className="text-sm flex-1">{ph.title}</Typography.Text>
                      <Tag color={ph.severity === "critical" ? "red" : ph.severity === "high" ? "orange" : "default"} className="!text-xs">
                        {ph.severity}
                      </Tag>
                    </div>
                  </List.Item>
                )}
              />
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}

export default CharacterWorldTab
