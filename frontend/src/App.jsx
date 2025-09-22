import { Navigate, Route, Routes } from 'react-router-dom'
import MainLayout from '@/layouts/MainLayout.jsx'
import DashboardPage from '@/pages/Dashboard.jsx'
import CorpusExplorerPage from '@/pages/CorpusExplorer.jsx'
import TranslationLabPage from '@/pages/TranslationLab.jsx'
import SummaryStudioPage from '@/pages/SummaryStudio.jsx'
import EvaluationBoardPage from '@/pages/EvaluationBoard.jsx'
import OperationsCenterPage from '@/pages/OperationsCenter.jsx'

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="corpus" element={<CorpusExplorerPage />} />
        <Route path="translation" element={<TranslationLabPage />} />
        <Route path="summaries" element={<SummaryStudioPage />} />
        <Route path="evaluation" element={<EvaluationBoardPage />} />
        <Route path="operations" element={<OperationsCenterPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default App
