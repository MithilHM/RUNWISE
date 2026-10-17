import { createContext, useContext } from 'react'
import type { ProjectResponse } from '../api/types'

export interface ProjectContextValue {
  projects: ProjectResponse[]
  projectId: string | null
  project: ProjectResponse | null
  isLoading: boolean
  error: unknown
  setProjectId: (id: string) => void
  refetchProjects: () => void
}

export const ProjectContext = createContext<ProjectContextValue | null>(null)

export function useProject(): ProjectContextValue {
  const context = useContext(ProjectContext)
  if (!context) throw new Error('useProject must be used inside a ProjectProvider')
  return context
}
