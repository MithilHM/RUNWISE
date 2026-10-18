import { Suspense, lazy } from 'react'
import { Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { ProjectProvider } from './components/ProjectContext'
import { LoadingState } from './components/States'
import { ShowcaseHeroPage } from './pages/ShowcaseHeroPage'
import { WorkspacePage } from './pages/WorkspacePage'

// The lineage screen pulls in the force-graph renderer and the copilot pulls in the
// chart stack. Neither is needed to open the workspace, so both are split out.
// The runtime screen ships the SSE reader and the execution-tree renderer with it.
const CopilotPage = lazy(() => import('./pages/CopilotPage').then((m) => ({ default: m.CopilotPage })))
const LineagePage = lazy(() => import('./pages/LineagePage').then((m) => ({ default: m.LineagePage })))
const RuntimePage = lazy(() => import('./pages/RuntimePage').then((m) => ({ default: m.RuntimePage })))

export default function App() {
  return (
    <ProjectProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<ShowcaseHeroPage />} />
          <Route path="workspace" element={<WorkspacePage />} />
          <Route
            path="copilot"
            element={
              <Suspense fallback={<LoadingState label="Loading copilot" />}>
                <CopilotPage />
              </Suspense>
            }
          />
          <Route
            path="lineage"
            element={
              <Suspense fallback={<LoadingState label="Loading graph" />}>
                <LineagePage />
              </Suspense>
            }
          />
          <Route
            path="runtime"
            element={
              <Suspense fallback={<LoadingState label="Loading autonomous runtime" />}>
                <RuntimePage />
              </Suspense>
            }
          />
        </Route>
      </Routes>
    </ProjectProvider>
  )
}
