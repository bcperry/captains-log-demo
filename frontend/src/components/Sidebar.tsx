import { SUPPORTED_LANGUAGES } from '../types/layout'
import type { SidebarProps } from '../types/layout'

export function Sidebar({
  isOpen,
  onClose,
  selectedLanguage,
  onLanguageChange,
  speechStatus,
  openAIStatus,
  region,
  endpoint,
  version,
  onTestSpeech,
  onTestOpenAI,
}: SidebarProps) {
  // Mask endpoint for display
  const maskedEndpoint =
    endpoint && endpoint.length > 30
      ? `${endpoint.slice(0, 25)}...${endpoint.slice(-15)}`
      : endpoint

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 h-full w-80 bg-white shadow-xl z-50 transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:shadow-none lg:z-auto ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="h-full flex flex-col overflow-y-auto">
          {/* Sidebar header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 lg:hidden">
            <h2 className="text-lg font-semibold text-gray-800">⚙️ Configuration</h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              aria-label="Close sidebar"
            >
              <svg
                className="w-5 h-5 text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          {/* Desktop header */}
          <div className="hidden lg:block p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-800">⚙️ Configuration</h2>
          </div>

          {/* Sidebar content */}
          <div className="flex-1 p-4 space-y-6">
            {/* Language selection */}
            <div>
              <label
                htmlFor="language-select"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Speech Language
              </label>
              <select
                id="language-select"
                value={selectedLanguage}
                onChange={(e) => onLanguageChange(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-lg bg-white text-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-gray-500">
                Select the primary language spoken in your audio file
              </p>
            </div>

            <hr className="border-gray-200" />

            {/* Service Status */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-3">🔗 Service Status</h3>

              {/* Azure Speech status */}
              <div
                className={`p-3 rounded-lg mb-2 ${
                  speechStatus.connected
                    ? 'bg-green-50 border border-green-200'
                    : 'bg-red-50 border border-red-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span>{speechStatus.connected ? '✅' : '❌'}</span>
                  <span
                    className={`text-sm font-medium ${
                      speechStatus.connected ? 'text-green-800' : 'text-red-800'
                    }`}
                  >
                    {speechStatus.name}
                  </span>
                </div>
                {speechStatus.message && (
                  <p
                    className={`text-xs mt-1 ${
                      speechStatus.connected ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {speechStatus.message}
                  </p>
                )}
              </div>

              {/* Azure OpenAI status */}
              <div
                className={`p-3 rounded-lg ${
                  openAIStatus.connected
                    ? 'bg-green-50 border border-green-200'
                    : 'bg-red-50 border border-red-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span>{openAIStatus.connected ? '✅' : '❌'}</span>
                  <span
                    className={`text-sm font-medium ${
                      openAIStatus.connected ? 'text-green-800' : 'text-red-800'
                    }`}
                  >
                    {openAIStatus.name}
                  </span>
                </div>
                {openAIStatus.message && (
                  <p
                    className={`text-xs mt-1 ${
                      openAIStatus.connected ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {openAIStatus.message}
                  </p>
                )}
              </div>
            </div>

            {/* Environment info */}
            {(region || maskedEndpoint) && (
              <div className="space-y-2">
                {region && (
                  <div className="p-2 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-xs text-blue-800">
                      <span className="font-medium">Region:</span> {region}
                    </p>
                  </div>
                )}
                {maskedEndpoint && (
                  <div className="p-2 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-xs text-blue-800 break-all">
                      <span className="font-medium">Endpoint:</span> {maskedEndpoint}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Test connection buttons */}
            <div className="grid grid-cols-2 gap-2">
              {onTestSpeech && (
                <button
                  onClick={onTestSpeech}
                  className="px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  🎤 Test Speech
                </button>
              )}
              {onTestOpenAI && (
                <button
                  onClick={onTestOpenAI}
                  className="px-3 py-2 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                >
                  🤖 Test OpenAI
                </button>
              )}
            </div>
          </div>

          {/* Footer with version */}
          <div className="p-4 border-t border-gray-200">
            <p className="text-xs text-gray-500 text-center">
              Captain&apos;s Log Version: {version}
            </p>
          </div>
        </div>
      </aside>
    </>
  )
}
