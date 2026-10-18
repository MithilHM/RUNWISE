import { NavLink, Outlet } from 'react-router-dom'
import { useHealth } from '../api/hooks'
import { useProject } from '../lib/projectContext'
import { DemoModeBadge, ErrorState, LoadingState } from './States'

const NAV = [
  { to: '/', label: '⚡ Hackathon Showcase', hint: 'Dataset & Lineage Story' },
  { to: '/workspace', label: 'Experiment workspace', hint: 'Screen A: All Runs Table' },
  { to: '/copilot', label: 'Research copilot', hint: 'Screen B: Custom Queries' },
  { to: '/lineage', label: 'Lineage & impact', hint: 'Screen C: Raw Graph & Queue' },
]

export function AppShell() {
  const { projects, projectId, setProjectId, isLoading, error, refetchProjects } = useProject()
  const { data: health } = useHealth()
  const graphUp = health?.database?.status === 'healthy'

  return (
    <div className="flex min-h-full flex-col bg-ink-950">
      <header className="border-b border-ink-700 bg-ink-900">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-4 px-6 py-3">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold tracking-[0.18em] text-ink-100">RUNWISE</span>
            <span className="text-xs text-ink-500">research experiment copilot</span>
          </div>

          <nav className="flex items-center gap-1" aria-label="Primary">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `rounded-md px-3 py-1.5 text-sm transition ${
                    isActive
                      ? 'bg-ink-800 text-ink-100'
                      : 'text-ink-400 hover:bg-ink-850 hover:text-ink-100'
                  }`
                }
                title={item.hint}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <DemoModeBadge llmConfigured={null} />
            <span
              className={`flex items-center gap-1.5 font-mono text-[11px] ${
                graphUp ? 'text-good-400' : 'text-danger-400'
              }`}
              title={graphUp ? `FalkorDB reachable: ${health?.database.graph}` : 'FalkorDB is not reachable'}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${graphUp ? 'bg-good-400' : 'bg-danger-400'}`} />
              {graphUp ? 'graph' : 'graph down'}
            </span>

            {isLoading ? (
              <span className="text-xs text-ink-500">loading projects…</span>
            ) : (
              <label className="flex items-center gap-2 text-xs text-ink-400">
                <span className="label">Project</span>
                <select
                  className="input w-56 py-1"
                  value={projectId ?? ''}
                  onChange={(event) => setProjectId(event.target.value)}
                  disabled={!projects.length}
                >
                  {projects.length === 0 && <option value="">no projects</option>}
                  {projects.map((project) => (
                    <option key={project.id} value={project.id}>
                      {project.name} ({project.experiment_count})
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1600px] flex-1 px-6 py-6">
        {error ? (
          <ErrorState error={error} onRetry={refetchProjects} context="Could not load projects" />
        ) : isLoading ? (
          <LoadingState label="Loading projects" />
        ) : (
          <Outlet />
        )}
      </main>

      <footer className="border-t border-ink-800 px-6 py-3 text-[11px] text-ink-600">
        Scores are documented heuristics, not probabilities. Records under an open correction are withheld, never
        deleted.
      </footer>
    </div>
  )
}
