import { Suspense, lazy } from 'react'
import { Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { ProjectProvider } from './components/ProjectContext'
import { LoadingState } from './components/States'
import { WorkspacePage } from './pages/WorkspacePage'

// The lineage screen pulls in the force-graph renderer and the copilot pulls in the
// chart stack. Neither is needed to open the workspace, so both are split out.
const CopilotPage = lazy(() => import('./pages/CopilotPage').then((m) => ({ default: m.CopilotPage })))
const LineagePage = lazy(() => import('./pages/LineagePage').then((m) => ({ default: m.LineagePage })))

export default function App() {
  return (
    <ProjectProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<WorkspacePage />} />
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
        </Route>
      </Routes>
    </ProjectProvider>
  )
}
