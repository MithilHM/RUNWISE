import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'

/** Whether an error is worth retrying at all.
 *
 * A 503 from a downed FalkorDB will still be a 503 in two seconds, so the
 * backend's "the graph is not reachable" envelope is treated as final and
 * surfaced to the user immediately rather than hidden behind a spinner that
 * never resolves.
 */
function isTransient(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) return true
  const detail = (error as { detail?: Array<{ code?: string }> }).detail
  if (!Array.isArray(detail)) return true
  const codes = detail.map((d) => d?.code)
  return codes.includes('graph_unavailable') || codes.includes('service_unavailable')
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error) => {
        if (!isTransient(error)) return false
        return failureCount < 1
      },
      refetchOnWindowFocus: false,
      staleTime: 10_000,
    },
  },
})

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)