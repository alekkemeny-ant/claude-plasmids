import { test as base, expect, Page } from '@playwright/test';

/**
 * Shared fixtures for Plasmid Designer browser tests.
 */

export const HAS_API_KEY = !!process.env.ANTHROPIC_API_KEY;

/** Skip a test when no real API key is available. */
export function requireApiKey() {
  base.skip(
    !HAS_API_KEY,
    'Requires ANTHROPIC_API_KEY — skipping live agent test',
  );
}

type Fixtures = {
  /** Page preloaded at `/` with sessionStorage cleared and health confirmed online. */
  freshPage: Page;
  /** Waits for the health badge to read "Agent Online". */
  waitForHealthy: (page: Page) => Promise<void>;
  /** Waits for an in-flight SSE stream to finish (send visible, stop hidden). */
  waitForStreamDone: (page: Page, timeout?: number) => Promise<void>;
};

export const test = base.extend<Fixtures>({
  waitForHealthy: async ({}, use) => {
    await use(async (page: Page) => {
      await expect(page.locator('#health-text')).toContainText('Agent Online', {
        timeout: 15_000,
      });
      await expect(page.locator('#health-badge')).not.toHaveClass(/offline/);
    });
  },

  waitForStreamDone: async ({}, use) => {
    await use(async (page: Page, timeout = 120_000) => {
      await expect(page.locator('#send-btn')).toBeVisible({ timeout });
      await expect(page.locator('#stop-btn')).toBeHidden();
    });
  },

  freshPage: async ({ page, request, waitForHealthy }, use) => {
    // Wipe server-side sessions so tests that inspect the sidebar session
    // list don't see entries created by prior tests in the same run.
    await request.post('/api/reset');
    // Clear client-side session state before each test so we always start from
    // the welcome screen.
    await page.addInitScript(() => {
      try {
        sessionStorage.clear();
      } catch {
        /* ignore — storage may not be available before navigation */
      }
    });
    await page.goto('/');
    await waitForHealthy(page);
    await use(page);
  },
});

export { expect };
