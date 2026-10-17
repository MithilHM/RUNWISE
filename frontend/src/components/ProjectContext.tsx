import { useCallback, useMemo, useState, type ReactNode } from 'react'
import { useProjects } from '../api/hooks'
import { ProjectContext, type ProjectContextValue } from '../lib/projectContext'

const STORAGE_KEY = 'runwise.projectId'

function readStoredProject(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export function ProjectProvider({ children }: { children: ReactNode }) {
  const { data, isLoading, error, refetch } = useProjects()
  const [projectId, setProjectIdState] = useState<string | null>(() => readStoredProject())

  // A stored project that no longer exists must not leave the UI pointing at a
  // project the backend cannot serve.
  const resolvedId = useMemo(() => {
    if (!data?.length) return null
    if (projectId && data.some((p) => p.id === projectId)) return projectId
    return data[0].id
  }, [data, projectId])

  const setProjectId = useCallback((id: string) => {
    setProjectIdState(id)
    try {
      window.localStorage.setItem(STORAGE_KEY, id)
    } catch {
      /* private browsing: the selection simply does not persist */
    }
  }, [])

  const value = useMemo<ProjectContextValue>(
    () => ({
      projects: data ?? [],
      projectId: resolvedId,
      project: data?.find((p) => p.id === resolvedId) ?? null,
      isLoading,
      error,
      setProjectId,
      refetchProjects: () => void refetch(),
    }),
    [data, resolvedId, isLoading, error, setProjectId, refetch],
  )

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>
}
