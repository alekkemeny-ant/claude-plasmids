import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for the Plasmid Designer web UI.
 *
 * Starts the Python server automatically via webServer. Tests that exercise
 * the live agent loop require ANTHROPIC_API_KEY to be set in the environment;
 * they will be skipped otherwise.
 */
export default defineConfig({
  testDir: './tests/browser',
  timeout: 180_000, // agent-loop tests can take a while
  expect: {
    timeout: 15_000,
  },
  fullyParallel: false, // sessions.json is a shared file — serialize to avoid races
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://localhost:8000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'python app/app.py --port 8000',
    url: 'http://localhost:8000/api/health',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
    stdout: 'pipe',
    stderr: 'pipe',
    env: {
      // Pass through the API key (if set) so live-agent tests can run.
      ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY ?? '',
      // Isolate the sessions file so tests don't clobber real user data.
      PLASMID_SESSIONS_PATH:
        process.env.PLASMID_SESSIONS_PATH ?? 'app/.sessions.playwright.json',
    },
  },
});
