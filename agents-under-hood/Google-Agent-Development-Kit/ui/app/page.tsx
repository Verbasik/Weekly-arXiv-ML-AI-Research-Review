'use client';

import { useState } from 'react';
import { FolderOpen, GitCommit, GitBranch, Activity, Settings } from 'lucide-react';

interface AnalysisResult {
  status: any;
  changes: any[];
  analysis: string;
  summary: any;
  timestamp: string;
}

interface ProjectStatus {
  project_path: string;
  git_status: any;
  recent_commits: any[];
  timestamp: string;
}

export default function Home() {
  const [projectPath, setProjectPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [projectStatus, setProjectStatus] = useState<ProjectStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  const analyzeChanges = async () => {
    if (!projectPath.trim()) {
      setError('Укажите путь к проекту');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ project_path: projectPath }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Ошибка при анализе изменений');
      }

      setAnalysisResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Неизвестная ошибка');
    } finally {
      setIsLoading(false);
    }
  };

  const getProjectStatus = async () => {
    if (!projectPath.trim()) {
      setError('Укажите путь к проекту');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/status/${encodeURIComponent(projectPath)}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Ошибка при получении статуса');
      }

      setProjectStatus(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Неизвестная ошибка');
    } finally {
      setIsLoading(false);
    }
  };

  const autoCommitAndPush = async () => {
    if (!projectPath.trim()) {
      setError('Укажите путь к проекту');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/auto-commit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ project_path: projectPath }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Ошибка при автоматическом коммите');
      }

      // Обновляем статус после коммита
      await getProjectStatus();
      
      setAnalysisResult(data.analysis || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Неизвестная ошибка');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Git Workflow Agent
          </h1>
          <p className="text-lg text-gray-600">
            Автоматический анализ изменений и создание коммитов с помощью ИИ
          </p>
        </div>

        {/* Main Form */}
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex items-center space-x-4 mb-6">
              <FolderOpen className="w-6 h-6 text-blue-500" />
              <div className="flex-1">
                <label htmlFor="projectPath" className="block text-sm font-medium text-gray-700 mb-2">
                  Путь к проекту
                </label>
                <input
                  type="text"
                  id="projectPath"
                  value={projectPath}
                  onChange={(e) => setProjectPath(e.target.value)}
                  placeholder="/path/to/your/project"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            <div className="flex space-x-4">
              <button
                onClick={analyzeChanges}
                disabled={isLoading}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50"
              >
                <Activity className="w-4 h-4" />
                <span>Анализировать изменения</span>
              </button>

              <button
                onClick={getProjectStatus}
                disabled={isLoading}
                className="flex items-center space-x-2 px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
              >
                <GitBranch className="w-4 h-4" />
                <span>Статус проекта</span>
              </button>

              <button
                onClick={autoCommitAndPush}
                disabled={isLoading}
                className="flex items-center space-x-2 px-4 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 disabled:opacity-50"
              >
                <GitCommit className="w-4 h-4" />
                <span>Авто-коммит и пуш</span>
              </button>
            </div>

            {isLoading && (
              <div className="mt-4 text-center">
                <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                <span className="ml-2 text-gray-600">Обработка...</span>
              </div>
            )}

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
                <p className="text-red-800">{error}</p>
              </div>
            )}
          </div>

          {/* Results */}
          {analysisResult && (
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Результат анализа</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Статус Git</h3>
                  <div className="bg-gray-50 p-4 rounded-md">
                    {analysisResult.status ? (
                      <>
                        <p><strong>Ветка:</strong> {analysisResult.status.branch || 'N/A'}</p>
                        <p><strong>Файлов в индексе:</strong> {analysisResult.status.staged_files?.length || 0}</p>
                        <p><strong>Измененных файлов:</strong> {analysisResult.status.modified_files?.length || 0}</p>
                        <p><strong>Новых файлов:</strong> {analysisResult.status.untracked_files?.length || 0}</p>
                      </>
                    ) : (
                      <p className="text-gray-600">Статус Git недоступен</p>
                    )}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Сводка изменений</h3>
                  <div className="bg-gray-50 p-4 rounded-md">
                    {analysisResult.summary ? (
                      <>
                        <p><strong>Всего изменений:</strong> {analysisResult.summary.total_changes || 0}</p>
                        <p><strong>По типам:</strong></p>
                        <ul className="list-disc list-inside ml-4">
                          {analysisResult.summary.by_type && Object.entries(analysisResult.summary.by_type).map(([type, count]) => (
                            <li key={type}>{type}: {count}</li>
                          ))}
                        </ul>
                        {analysisResult.summary.by_language && Object.keys(analysisResult.summary.by_language).length > 0 && (
                          <>
                            <p><strong>По языкам программирования:</strong></p>
                            <ul className="list-disc list-inside ml-4">
                              {Object.entries(analysisResult.summary.by_language).map(([lang, count]) => (
                                <li key={lang}>{lang}: {count}</li>
                              ))}
                            </ul>
                          </>
                        )}
                      </>
                    ) : (
                      <p className="text-gray-600">Сводка изменений недоступна</p>
                    )}
                  </div>
                </div>
              </div>

              <div className="mt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Предложенное сообщение коммита</h3>
                <div className="bg-gray-50 p-4 rounded-md">
                  <pre className="whitespace-pre-wrap text-sm">{analysisResult.analysis || 'Анализ недоступен'}</pre>
                </div>
              </div>

              <div className="mt-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Измененные файлы</h3>
                <div className="space-y-2">
                  {analysisResult.changes && analysisResult.changes.length > 0 ? (
                    analysisResult.changes.map((change, index) => (
                      <div key={index} className="flex items-center space-x-4 p-3 bg-gray-50 rounded-md">
                        <span className="text-sm font-medium">{change.file_path || 'Неизвестный файл'}</span>
                        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                          {change.change_type || 'неизвестно'}
                        </span>
                        {change.language && (
                          <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                            {change.language}
                          </span>
                        )}
                      </div>
                    ))
                  ) : (
                    <p className="text-gray-600">Нет измененных файлов</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {projectStatus && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Статус проекта</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Git статус</h3>
                  <div className="bg-gray-50 p-4 rounded-md">
                    <p><strong>Путь:</strong> {projectStatus.project_path}</p>
                    <p><strong>Ветка:</strong> {projectStatus.git_status.branch || 'N/A'}</p>
                    <p><strong>Удаленный репозиторий:</strong> {projectStatus.git_status.remote || 'N/A'}</p>
                    <p><strong>Ahead:</strong> {projectStatus.git_status.ahead}</p>
                    <p><strong>Behind:</strong> {projectStatus.git_status.behind}</p>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Последние коммиты</h3>
                  <div className="space-y-2">
                    {projectStatus.recent_commits.map((commit, index) => (
                      <div key={index} className="bg-gray-50 p-3 rounded-md">
                        <p className="text-sm font-medium">{commit.message}</p>
                        <p className="text-xs text-gray-600">
                          {commit.author} • {new Date(commit.date).toLocaleString()}
                        </p>
                        <p className="text-xs text-gray-500 font-mono">{commit.hash.substring(0, 8)}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 