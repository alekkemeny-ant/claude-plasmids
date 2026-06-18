/**
 * End-to-end browser tests for the Plasmid Designer web UI.
 *
 * Run with:  npx playwright test
 *
 * Tests marked `requireApiKey()` exercise the live agent loop and will be
 * skipped when ANTHROPIC_API_KEY is unset.
 *
 * data-testid suggestions (the current HTML relies on classes/ids only):
 *   #health-badge            -> data-testid="health-badge"
 *   #send-btn                -> data-testid="send-btn"
 *   #stop-btn                -> data-testid="stop-btn"
 *   .new-chat-btn            -> data-testid="new-chat-btn"
 *   .session-item            -> data-testid="session-item"
 *   .session-item .delete-btn-> data-testid="session-delete-btn"
 *   .tool-block .block-label -> data-testid="tool-label"
 *   .thinking-block          -> data-testid="thinking-block"
 *   .batch-card              -> data-testid="batch-card"
 *   .batch-dl-all-btn        -> data-testid="batch-download-all"
 *   #model-select            -> data-testid="model-select"
 *   #drop-overlay            -> data-testid="drop-overlay"
 *   .example-btn             -> data-testid="example-btn"
 */

import { test, expect, requireApiKey } from './fixtures';
import type { Page } from '@playwright/test';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Dispatch a `drop` event carrying a CSV file on #chat-panel. */
async function dropCsv(page: Page, csv: string, filename = 'batch.csv') {
  const dataTransfer = await page.evaluateHandle(
    ({ csv, filename }) => {
      const dt = new DataTransfer();
      dt.items.add(new File([csv], filename, { type: 'text/csv' }));
      return dt;
    },
    { csv, filename },
  );
  await page.dispatchEvent('#chat-panel', 'drop', { dataTransfer });
}

/** Send a message and wait for the SSE stream to finish. */
async function sendAndWait(
  page: Page,
  text: string,
  timeout = 120_000,
): Promise<void> {
  await page.locator('#input').fill(text);
  await page.locator('#send-btn').click();
  await expect(page.locator('#stop-btn')).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('#send-btn')).toBeVisible({ timeout });
  await expect(page.locator('#stop-btn')).toBeHidden();
}

// Mocked SSE body that immediately ends the stream.
const MOCK_SSE_DONE =
  'data: {"type":"session","session_id":"mock-sess"}\n\n' +
  'data: {"type":"done"}\n\n';

// ===========================================================================
// chat-flow
// ===========================================================================
test.describe('chat-flow', () => {
  test('Quick-start button triggers full SSE stream with thinking, tool calls, and response', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await expect(page.locator('#welcome')).toBeVisible();
    await expect(page.locator('#health-text')).toContainText('Agent Online');

    // Suggestion: add data-testid="example-btn" to .examples button
    await page
      .locator('.examples button', { hasText: 'What backbones are available?' })
      .click();

    await expect(page.locator('#welcome')).toBeHidden();
    await expect(page.locator('#stop-btn')).toBeVisible();

    await expect(page.locator('.thinking-block').first()).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.locator('.tool-block').first()).toBeVisible({
      timeout: 60_000,
    });
    await expect(
      page.locator('.msg.assistant .msg-bubble-assistant').first(),
    ).toBeVisible({ timeout: 60_000 });

    // Stream complete
    await expect(page.locator('#stop-btn')).toBeHidden({ timeout: 60_000 });
    await expect(page.locator('#send-btn')).toBeVisible();

    // Assertions
    await expect(page.locator('.msg.user .msg-bubble-user')).toContainText(
      'What backbones are available?',
    );
    expect(await page.locator('.thinking-block').count()).toBeGreaterThanOrEqual(1);
    expect(await page.locator('.tool-block').count()).toBeGreaterThanOrEqual(1);
    await expect(
      page.locator('.tool-block .block-header').first(),
    ).toContainText(/backbone/i);

    const assistantText = await page
      .locator('.msg-bubble-assistant')
      .last()
      .textContent();
    expect(assistantText ?? '').not.toBe('');
    expect(assistantText ?? '').toMatch(/pcDNA|backbone/i);

    await expect(page.locator('#input')).toBeEnabled();
    await expect(page.locator('#input')).toHaveValue('');
    await expect(page.locator('#send-btn')).toBeVisible();
    await expect(page.locator('#stop-btn')).toBeHidden();

    const sessionId = await page.evaluate(() =>
      sessionStorage.getItem('plasmid_session_id'),
    );
    expect(sessionId).toBeTruthy();

    await expect(page.locator('.streaming-cursor')).toHaveCount(0);
  });

  test('Typed input + Enter key sends message and streams assistant response', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    const chatRequests: Request[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/chat') && req.method() === 'POST') {
        chatRequests.push(req);
      }
    });

    await page.locator('#input').click();
    await page
      .locator('#input')
      .type('Design an EGFP expression plasmid using pcDNA3.1(+)');
    await page.locator('#input').press('Enter');

    await expect(page.locator('#welcome')).toBeHidden();
    await expect(page.locator('#input')).toBeDisabled();
    await expect(page.locator('.msg.user')).toBeVisible();
    await expect(page.locator('.tool-block').first()).toBeVisible({
      timeout: 60_000,
    });
    await expect(page.locator('.msg-bubble-assistant').first()).toBeVisible({
      timeout: 120_000,
    });
    await expect(page.locator('#send-btn')).toBeVisible({ timeout: 120_000 });

    // Assertions
    await expect(page.locator('.msg-bubble-user')).toHaveText(
      'Design an EGFP expression plasmid using pcDNA3.1(+)',
    );
    await expect(page.locator('#input')).toHaveValue('');
    expect(await page.locator('.tool-block').count()).toBeGreaterThanOrEqual(2);
    await expect(page.locator('.msg-bubble-assistant').last()).toContainText(
      /EGFP|pcDNA|assembl/i,
    );
    await expect(page.locator('#input')).toBeEnabled();
    await expect(page.locator('#input')).toBeFocused();
    expect(
      await page.locator('#sessions-list .session-item').count(),
    ).toBeGreaterThanOrEqual(1);

    // Network: exactly one POST to /api/chat, SSE response
    expect(chatRequests).toHaveLength(1);
    expect(chatRequests[0].headers()['content-type']).toContain(
      'application/json',
    );
    const resp = await chatRequests[0].response();
    expect(resp?.headers()['content-type']).toContain('text/event-stream');
  });

  test('Stop button aborts in-flight SSE stream and restores input state', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    const consoleErrors: string[] = [];
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    let sawCancel = false;
    page.on('request', (req) => {
      if (req.url().match(/\/api\/sessions\/[^/]+\/cancel/)) sawCancel = true;
    });

    await page
      .locator('.examples button', {
        hasText: 'Assemble tdTomato in pcDNA3.1(+) and export as GenBank',
      })
      .click();

    await expect(page.locator('#stop-btn')).toBeVisible();
    // Confirm stream is mid-flight before stopping
    await page
      .locator('.thinking-block, .tool-block')
      .first()
      .waitFor({ state: 'visible', timeout: 60_000 });

    await page.locator('#stop-btn').click();
    await expect(page.locator('#send-btn')).toBeVisible({ timeout: 5_000 });

    await expect(page.locator('#stop-btn')).toBeHidden();
    await expect(page.locator('#input')).toBeEnabled();
    expect(sawCancel).toBe(true);
    await expect(page.locator('.streaming-cursor')).toHaveCount(0);
    await expect(page.locator('.msg.user')).toHaveCount(1);
    // AbortError should be swallowed by name check
    expect(
      consoleErrors.filter((e) => e.includes('AbortError')),
    ).toHaveLength(0);
  });

  test('Shift+Enter inserts newline without sending; empty message does not send', async ({
    freshPage: page,
  }) => {
    const chatCalls: string[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/chat')) chatCalls.push(req.url());
    });

    const input = page.locator('#input');
    await input.focus();
    await input.type('line one');
    const heightBefore = await input.evaluate(
      (el: HTMLTextAreaElement) => el.offsetHeight,
    );
    await input.press('Shift+Enter');
    await input.type('line two');

    await expect(input).toHaveValue('line one\nline two');
    await expect(page.locator('.msg.user')).toHaveCount(0);
    await expect(page.locator('#welcome')).toBeVisible();
    const heightAfter = await input.evaluate(
      (el: HTMLTextAreaElement) => el.offsetHeight,
    );
    expect(heightAfter).toBeGreaterThanOrEqual(heightBefore);

    // Whitespace only
    await input.fill('   ');
    await input.press('Enter');
    await page.waitForTimeout(500);
    expect(chatCalls).toHaveLength(0);
    await expect(page.locator('#welcome')).toBeVisible();

    // Empty + click send
    await input.fill('');
    await page.locator('#send-btn').click();
    await page.waitForTimeout(300);
    expect(chatCalls).toHaveLength(0);
    await expect(page.locator('#send-btn')).toBeVisible();
    await expect(page.locator('#stop-btn')).toBeHidden();
    await expect(input).toBeEnabled();
  });

  test('Session persists across reload — messages restored from /api/sessions/{id}/messages', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await page
      .locator('.examples button', {
        hasText: 'Put mCherry into a mammalian expression vector',
      })
      .click();
    await expect(page.locator('#send-btn')).toBeVisible({ timeout: 120_000 });

    const sessionId = await page.evaluate(() =>
      sessionStorage.getItem('plasmid_session_id'),
    );
    expect(sessionId).toBeTruthy();
    const responseText = await page
      .locator('.msg-bubble-assistant')
      .last()
      .textContent();

    const messagesResp = page.waitForResponse(
      (r) =>
        r.url().includes(`/api/sessions/${sessionId}/messages`) &&
        r.status() === 200,
    );
    await page.reload();
    await messagesResp;

    await expect(page.locator('#messages .msg').first()).toBeVisible({
      timeout: 10_000,
    });

    const sessionIdAfter = await page.evaluate(() =>
      sessionStorage.getItem('plasmid_session_id'),
    );
    expect(sessionIdAfter).toBe(sessionId);
    await expect(page.locator('#welcome')).toBeHidden();
    await expect(page.locator('.msg-bubble-user')).toContainText('mCherry');
    const restored = await page
      .locator('.msg-bubble-assistant')
      .last()
      .textContent();
    expect(restored ?? '').not.toBe('');
    expect(restored).toBe(responseText);
    expect(
      await page.locator('.thinking-block, .tool-block').count(),
    ).toBeGreaterThanOrEqual(1);
    // Suggestion: add data-testid="session-item-active" or data-session-id attr
    await expect(
      page.locator('#sessions-list .session-item.active'),
    ).toHaveCount(1);
  });
});

// ===========================================================================
// session-mgmt
// ===========================================================================
test.describe('session-mgmt', () => {
  test('Create new session via chat and verify sidebar appearance', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await expect(page.locator('#welcome')).toBeVisible();
    const initialCount = await page
      .locator('#sessions-list .session-item')
      .count();

    await sendAndWait(page, 'What backbones are available?');

    // Force sidebar refresh rather than waiting for the 5s poll
    await page.evaluate(() => (window as any).loadSessions?.());
    await expect(
      page.locator('#sessions-list .session-item').first(),
    ).toBeVisible({ timeout: 10_000 });

    const newCount = await page.locator('#sessions-list .session-item').count();
    expect(newCount).toBe(initialCount + 1);

    const activeItem = page.locator('#sessions-list .session-item.active');
    await expect(activeItem).toHaveCount(1);
    await expect(activeItem.locator('.session-name')).toContainText(
      'What backbones are available?',
    );
    await expect(page.locator('#welcome')).toBeHidden();
    await expect(page.locator('#messages .msg.user')).toContainText(
      'What backbones are available?',
    );
  });

  test('New Chat button clears active session and resets to welcome screen', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await sendAndWait(page, 'List available inserts');
    await page.evaluate(() => (window as any).loadSessions?.());
    await expect(
      page.locator('#sessions-list .session-item.active'),
    ).toBeVisible();

    await page.locator('.new-chat-btn').click();
    await expect(page.locator('#welcome')).toBeVisible();

    await expect(page.locator('#welcome h1, #welcome h2')).toContainText(
      /Design an expression plasmid/i,
    );
    await expect(page.locator('#messages .msg')).toHaveCount(0);
    await expect(
      page.locator('#sessions-list .session-item.active'),
    ).toHaveCount(0);
    // Previous session still exists
    expect(
      await page.locator('#sessions-list .session-item').count(),
    ).toBeGreaterThanOrEqual(1);
    await expect(page.locator('#input')).toBeFocused();
    await expect(page.locator('#input')).toHaveValue('');
    const sid = await page.evaluate(() => sessionStorage.getItem('plasmid_session_id'));
    expect(sid).toBeNull();
  });

  test('Switch between two sessions and verify message history restoration', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await sendAndWait(
      page,
      'Design an EGFP expression plasmid using pcDNA3.1(+)',
    );
    await page.locator('.new-chat-btn').click();
    await sendAndWait(page, 'Put mCherry into a mammalian expression vector');

    await page.evaluate(() => (window as any).loadSessions?.());
    await expect(page.locator('#sessions-list .session-item')).toHaveCount(2, {
      timeout: 10_000,
    });

    // Select Session A
    const sessA = page
      .locator('#sessions-list .session-item')
      .filter({ hasText: 'Design an EGFP' });
    const respA = page.waitForResponse((r) =>
      r.url().includes('/messages'),
    );
    await sessA.click();
    await respA;

    await expect(sessA).toHaveClass(/active/);
    await expect(page.locator('#messages .msg.user')).toContainText(
      'Design an EGFP expression plasmid',
    );
    await expect(page.locator('#messages')).not.toContainText('mCherry');

    // Select Session B
    const sessB = page
      .locator('#sessions-list .session-item')
      .filter({ hasText: 'Put mCherry' });
    const respB = page.waitForResponse((r) =>
      r.url().includes('/messages'),
    );
    await sessB.click();
    await respB;

    await expect(sessB).toHaveClass(/active/);
    await expect(sessA).not.toHaveClass(/active/);
    await expect(page.locator('#messages .msg.user')).toContainText(
      'Put mCherry into a mammalian',
    );
    await expect(page.locator('#messages .msg.user')).not.toContainText('EGFP');
  });

  test('Delete active session returns to welcome screen and removes from sidebar', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await sendAndWait(page, 'What backbones are available?');
    await page.evaluate(() => (window as any).loadSessions?.());
    const active = page.locator('#sessions-list .session-item.active');
    await expect(active).toBeVisible();

    const beforeCount = await page
      .locator('#sessions-list .session-item')
      .count();
    const sessionId = await page.evaluate(
      () => sessionStorage.getItem('plasmid_session_id'),
    );

    const deleteResp = page.waitForResponse(
      (r) => r.url().includes('/api/sessions/') && r.request().method() === 'DELETE',
    );
    await active.hover();
    // Suggestion: add data-testid="session-delete-btn"
    await active.locator('.delete-btn').click();
    await deleteResp;

    await expect(page.locator('#sessions-list .session-item')).toHaveCount(
      beforeCount - 1,
    );
    if (beforeCount === 1) {
      await expect(page.locator('#sessions-list .no-sessions')).toContainText(
        'No conversations yet',
      );
    }
    await expect(page.locator('#welcome')).toBeVisible();
    await expect(page.locator('#messages .msg')).toHaveCount(0);
    const sid = await page.evaluate(() => sessionStorage.getItem('plasmid_session_id'));
    expect(sid).toBeNull();

    // Confirm backend dropped it
    const sessions = await page.evaluate(async () => {
      const r = await fetch('/api/sessions');
      return r.json();
    });
    expect(
      sessions.find((s: any) => s.session_id === sessionId),
    ).toBeUndefined();
  });

  test('Delete non-active session keeps current chat visible', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await sendAndWait(page, 'List available inserts');
    await page.locator('.new-chat-btn').click();
    await sendAndWait(page, 'What backbones are available?');

    await page.evaluate(() => (window as any).loadSessions?.());
    await expect(page.locator('#sessions-list .session-item')).toHaveCount(2);

    const sessionBId = await page.evaluate(
      () => sessionStorage.getItem('plasmid_session_id'),
    );

    const nonActive = page
      .locator('#sessions-list .session-item:not(.active)')
      .filter({ hasText: 'List available inserts' });
    await nonActive.hover();
    await nonActive.locator('.delete-btn').click();

    await expect(page.locator('#sessions-list .session-item')).toHaveCount(1);
    const remaining = page.locator('#sessions-list .session-item');
    await expect(remaining.locator('.session-name')).toContainText(
      'What backbones are available',
    );
    await expect(remaining).toHaveClass(/active/);
    await expect(page.locator('#welcome')).toBeHidden();
    await expect(page.locator('#messages .msg.user')).toContainText(
      'What backbones are available?',
    );
    const sidAfter = await page.evaluate(
      () => sessionStorage.getItem('plasmid_session_id'),
    );
    expect(sidAfter).toBe(sessionBId);
  });
});

// ===========================================================================
// golden-gate
// ===========================================================================
test.describe('golden-gate', () => {
  test('Golden Gate: single-part assembly invokes assemble_golden_gate and produces construct', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await page
      .locator('#input')
      .fill(
        'Perform a Golden Gate assembly using the AICS_X0001_pTwist_Kan_B backbone and the AICS_SynP000X part with Esp3I enzyme.',
      );
    await page.locator('#send-btn').click();
    await expect(page.locator('#stop-btn')).toBeVisible();

    const ggBlock = page
      .locator('.tool-block')
      .filter({ has: page.locator('.block-label', { hasText: 'assemble_golden_gate' }) });
    await expect(ggBlock.first()).toBeVisible({ timeout: 90_000 });

    await expect(page.locator('#send-btn')).toBeVisible({ timeout: 120_000 });
    await expect(page.locator('#stop-btn')).toBeHidden();

    await ggBlock.first().locator('.block-header').click();
    const body = ggBlock.first().locator('.block-body');
    await expect(body).toBeVisible();

    await expect(body).toContainText('Golden Gate assembly successful');
    await expect(body).toContainText('Esp3I');
    await expect(body).toContainText(/Total size\s*:\s*\d+ bp/);
    await expect(body).toContainText('Assembly order');
    // Pulse dot removed once tool finished
    await expect(ggBlock.first().locator('.pulse-dot')).toHaveCount(0);
    await expect(page.locator('.msg.assistant').last()).toContainText(
      /Golden Gate|assembled/i,
    );
  });

  test('Golden Gate: multi-part assembly orders parts by overhang matching', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await expect(page.locator('#welcome')).toBeVisible();
    await page
      .locator('#input')
      .fill(
        'Do a Golden Gate assembly with backbone AICS_X0001_pTwist_Kan_B and parts AICS_SynP000X, AICS_SynP000Y, and AICS_SynP000Z using Esp3I.',
      );
    await page.locator('#send-btn').click();

    const ggBlock = page
      .locator('.tool-block')
      .filter({ has: page.locator('.block-label', { hasText: 'assemble_golden_gate' }) });
    await expect(ggBlock.first()).toBeVisible({ timeout: 120_000 });
    await expect(page.locator('#send-btn')).toBeVisible({ timeout: 120_000 });

    await ggBlock.first().locator('.block-header').click();
    const body = ggBlock.first().locator('.block-body');

    const bodyText = (await body.textContent()) ?? '';
    expect(bodyText).toContain('AICS_X0001_pTwist_Kan_B');
    expect(bodyText).toContain('AICS_SynP000X');
    expect(bodyText).toContain('AICS_SynP000Y');
    expect(bodyText).toContain('AICS_SynP000Z');
    expect(bodyText).toContain('Golden Gate assembly successful');
    // Assembly order: 3 parts with arrows
    expect(bodyText).toMatch(/Assembly order.*→.*→/);
    // 4 junctions (backbone + 3 parts)
    const junctions = bodyText.match(/Junctions.*?:(.*)/)?.[1] ?? '';
    expect((junctions.match(/[ACGT]{4}/gi) ?? []).length).toBeGreaterThanOrEqual(4);
    expect(bodyText).toMatch(/Assembled sequence \(\d+ bp\):/);
    expect(bodyText).toMatch(/[ACGTacgt]{20,}/);
  });

  test('Golden Gate: SSE event sequence emits tool_use_start then tool_result for assemble_golden_gate', async ({
    request,
  }) => {
    requireApiKey();

    const resp = await request.post('/api/chat', {
      headers: { Accept: 'text/event-stream' },
      data: {
        message:
          'Golden Gate assemble AICS_SynP000X into AICS_X0001_pTwist_Kan_B with Esp3I',
      },
      timeout: 90_000,
    });

    expect(resp.headers()['content-type']).toContain('text/event-stream');
    const body = await resp.text();

    const events: any[] = [];
    for (const line of body.split('\n')) {
      if (line.startsWith('data: ')) {
        try {
          events.push(JSON.parse(line.slice(6)));
        } catch {
          /* skip malformed */
        }
      }
    }

    const startIdx = events.findIndex(
      (e) => e.type === 'tool_use_start' && e.tool === 'assemble_golden_gate',
    );
    expect(startIdx).toBeGreaterThanOrEqual(0);

    const resultIdx = events.findIndex(
      (e) => e.type === 'tool_result' && e.tool === 'assemble_golden_gate',
    );
    expect(resultIdx).toBeGreaterThan(startIdx);

    const result = events[resultIdx];
    expect(result.content).toContain('Golden Gate assembly successful');
    expect(result.input).toHaveProperty('backbone_id');
    expect(result.input).toHaveProperty('part_ids');
    expect(Array.isArray(result.input.part_ids)).toBe(true);
    expect(result.input.part_ids).toContain('AICS_SynP000X');

    expect(events[events.length - 1].type).toBe('done');
    expect(events.some((e) => e.type === 'error')).toBe(false);
  });

  test('Golden Gate: session persists assemble_golden_gate tool call in message history', async ({
    freshPage: page,
    request,
  }) => {
    requireApiKey();

    await sendAndWait(
      page,
      'Use Golden Gate to clone part AICS_SynP000Y into backbone AICS_X0001_pTwist_Kan_B with Esp3I enzyme.',
    );

    const sessionId = await page.evaluate(() =>
      sessionStorage.getItem('plasmid_session_id'),
    );
    expect(sessionId).toBeTruthy();

    await page.locator('.new-chat-btn').click();
    await expect(page.locator('#welcome')).toBeVisible();

    const msgResp = await request.get(`/api/sessions/${sessionId}/messages`);
    expect(msgResp.status()).toBe(200);
    const messages = await msgResp.json();
    expect(Array.isArray(messages)).toBe(true);

    const toolUse = messages
      .flatMap((m: any) =>
        Array.isArray(m.content) ? m.content : [],
      )
      .find(
        (b: any) => b.type === 'tool_use' && b.name === 'assemble_golden_gate',
      );
    expect(toolUse).toBeDefined();
    expect(toolUse.input.backbone_id).toBe('AICS_X0001_pTwist_Kan_B');
    expect(toolUse.input.part_ids).toContain('AICS_SynP000Y');

    // Restore via sidebar
    await page.evaluate(() => (window as any).loadSessions?.());
    await page
      .locator('#sessions-list .session-item')
      .filter({ hasText: 'Golden Gate' })
      .first()
      .click();

    const ggBlock = page
      .locator('.tool-block')
      .filter({ has: page.locator('.block-label', { hasText: 'assemble_golden_gate' }) });
    await expect(ggBlock.first()).toBeVisible();
    await ggBlock.first().locator('.block-header').click();
    await expect(ggBlock.first().locator('.block-body')).toContainText(
      'Golden Gate assembly successful',
    );
    await expect(
      page
        .locator('#sessions-list .session-item')
        .filter({ hasText: 'Golden Gate' }),
    ).toHaveCount(1);
  });

  test('Golden Gate: agent does NOT fall back to assemble_construct for GG requests', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    await sendAndWait(
      page,
      'I want to use Golden Gate cloning with Esp3I to insert the Allen Institute part AICS_SynP000Z into the AICS_X0001_pTwist_Kan_B destination vector.',
    );

    const labels = await page
      .locator('.tool-block .block-label')
      .allTextContents();

    expect(labels).toContain('assemble_golden_gate');
    expect(labels).not.toContain('assemble_construct');
    expect(labels).not.toContain('fuse_inserts');

    const ggBlock = page
      .locator('.tool-block')
      .filter({ has: page.locator('.block-label', { hasText: 'assemble_golden_gate' }) });
    await ggBlock.first().locator('.block-header').click();
    const bodyText =
      (await ggBlock.first().locator('.block-body').textContent()) ?? '';
    expect(bodyText).toMatch(/successful/i);
    expect(bodyText).toMatch(/\d+ bp/);
  });
});

// ===========================================================================
// csv-batch
// ===========================================================================
test.describe('csv-batch', () => {
  test('Drag-drop overlay appears and dismisses correctly', async ({
    freshPage: page,
  }) => {
    await expect(page.locator('#chat-panel')).toBeVisible();
    await expect(page.locator('#drop-overlay')).not.toHaveClass(/active/);

    // Simulate dragenter — app code checks dataTransfer.types.includes('Files')
    await page.evaluate(() => {
      const dt = new DataTransfer();
      dt.items.add(new File(['x'], 'x.csv', { type: 'text/csv' }));
      const ev = new DragEvent('dragenter', {
        bubbles: true,
        dataTransfer: dt,
      });
      document.getElementById('chat-panel')!.dispatchEvent(ev);
    });

    await expect(page.locator('#drop-overlay')).toHaveClass(/active/);
    await expect(page.locator('.drop-overlay-label')).toHaveText(
      'Drop CSV to batch design',
    );
    await expect(page.locator('.drop-overlay-sub')).toContainText(
      'Required column: description',
    );
    await expect(page.locator('.drop-overlay-sub')).toContainText(
      'Optional: name, output_format',
    );

    // dragleave
    await page.evaluate(() => {
      const dt = new DataTransfer();
      dt.items.add(new File(['x'], 'x.csv', { type: 'text/csv' }));
      const ev = new DragEvent('dragleave', {
        bubbles: true,
        dataTransfer: dt,
      });
      document.getElementById('chat-panel')!.dispatchEvent(ev);
    });
    await expect(page.locator('#drop-overlay')).not.toHaveClass(/active/);
  });

  test('CSV upload creates batch job and renders placeholder row cards', async ({
    freshPage: page,
  }) => {
    await expect(page.locator('#welcome')).toBeVisible();

    const csv =
      'description,name\n' +
      'EGFP in pcDNA3.1,egfp_construct\n' +
      'mCherry in pcDNA3.1,mcherry_construct\n' +
      'tdTomato in pcDNA3.1,tdtomato_construct\n';

    const respPromise = page.waitForResponse((r) =>
      r.url().includes('/api/batch') && r.request().method() === 'POST',
    );
    await dropCsv(page, csv, 'my_batch.csv');
    const resp = await respPromise;

    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.row_count).toBe(3);
    expect(typeof body.job_id).toBe('string');
    const jobId = body.job_id;

    await expect(page.locator('#welcome')).toBeHidden();
    const label = page.locator(`#batch-label-${jobId}`);
    await expect(label).toContainText('Batch designing 3 plasmids');
    await expect(label).toContainText('my_batch.csv');

    for (const i of [0, 1, 2]) {
      await expect(page.locator(`#batch-card-${jobId}-${i}`)).toBeVisible();
    }
    await expect(page.locator('.batch-card')).toHaveCount(3);
    await expect(
      page.locator('.batch-card .batch-row-meta').first(),
    ).toContainText(/Pending/i);
    await expect(page.locator('#drop-overlay')).not.toHaveClass(/active/);
  });

  test('Invalid CSV without description column is rejected with error', async ({
    freshPage: page,
  }) => {
    const csv = 'name,output_format\nfoo,genbank\nbar,fasta\n';

    let alertMsg = '';
    page.on('dialog', async (dialog) => {
      alertMsg = dialog.message();
      await dialog.accept();
    });

    const respPromise = page.waitForResponse((r) =>
      r.url().includes('/api/batch') && r.request().method() === 'POST',
    );
    await dropCsv(page, csv, 'bad.csv');
    const resp = await respPromise;

    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(JSON.stringify(body)).toMatch(/description/i);

    await expect.poll(() => alertMsg).toMatch(/Error.*description/i);
    await expect(page.locator('.batch-card')).toHaveCount(0);
    await expect(page.locator('#welcome')).toBeVisible();
  });

  test('Batch rows populate with descriptions and expand to show log on click', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    const csv =
      'description\nEGFP in pcDNA3.1(+)\nmCherry in pcDNA3.1(+)\n';

    const respPromise = page.waitForResponse(
      (r) => r.url().includes('/api/batch') && r.request().method() === 'POST',
    );
    await dropCsv(page, csv);
    const body = await (await respPromise).json();
    const jobId = body.job_id;

    // Wait for first poll
    await page.waitForResponse(
      (r) =>
        r.url().includes(`/api/batch/${jobId}`) &&
        r.request().method() === 'GET',
      { timeout: 5_000 },
    );

    const row0 = page.locator(`#batch-card-${jobId}-0`);
    const row1 = page.locator(`#batch-card-${jobId}-1`);
    await expect(row0.locator('.batch-row-desc')).toContainText(
      'EGFP in pcDNA3.1(+)',
    );
    await expect(row1.locator('.batch-row-desc')).toContainText(
      'mCherry in pcDNA3.1(+)',
    );
    await expect(row0.locator('.batch-row-meta')).toContainText(/^1 ·/);
    await expect(row1.locator('.batch-row-meta')).toContainText(/^2 ·/);

    const log0 = page.locator(`#batch-log-${jobId}-0`);
    const chev0 = page.locator(`#batch-chev-${jobId}-0`);
    await expect(log0).not.toHaveClass(/open/);

    await row0.locator('.batch-row-header').click();
    await expect(log0).toHaveClass(/open/);
    await expect(chev0).toHaveClass(/open/);

    await row0.locator('.batch-row-header').click();
    await expect(log0).not.toHaveClass(/open/);
  });

  test('Completed batch exposes per-row and Download All buttons that fetch files', async ({
    freshPage: page,
  }) => {
    requireApiKey();

    const csv =
      'description,output_format\nAssemble EGFP in pcDNA3.1(+),genbank\n';

    const respPromise = page.waitForResponse(
      (r) => r.url().includes('/api/batch') && r.request().method() === 'POST',
    );
    await dropCsv(page, csv);
    const body = await (await respPromise).json();
    const jobId = body.job_id;

    // Poll until done
    await page.waitForFunction(
      async (jid) => {
        const r = await fetch('/api/batch/' + jid);
        const j = await r.json();
        return (
          j.status === 'done' &&
          !j.rows?.some(
            (row: any) => row.status === 'pending' || row.status === 'running',
          )
        );
      },
      jobId,
      { timeout: 120_000, polling: 2_000 },
    );

    // Suggestion: add data-testid="batch-download-all"
    const dlAll = page.locator(`#batch-label-${jobId} .batch-dl-all-btn`);
    await expect(dlAll).toBeVisible({ timeout: 10_000 });
    await expect(dlAll).toContainText(/Download All.*\.zip/i);

    const rowDownloads = page.locator(
      `#batch-card-${jobId}-0 .batch-row-downloads .download-btn`,
    );
    await expect(rowDownloads.first()).toBeVisible();
    await expect(rowDownloads.first()).toContainText(/\.(gb|fasta)/i);

    // Per-row download
    const dl1Promise = page.waitForEvent('download');
    const dlRespPromise = page.waitForResponse((r) =>
      r.url().includes(`/api/batch/${jobId}/download/0/0`),
    );
    await rowDownloads.first().click();
    const dl1 = await dl1Promise;
    const dlResp = await dlRespPromise;
    expect(dlResp.status()).toBe(200);
    expect(dl1.suggestedFilename()).toMatch(/\.(gb|fasta)$/i);

    // Download all
    const dl2Promise = page.waitForEvent('download');
    const zipRespPromise = page.waitForResponse((r) =>
      r.url().includes(`/api/batch/${jobId}/download-all`),
    );
    await dlAll.click();
    const dl2 = await dl2Promise;
    const zipResp = await zipRespPromise;
    expect(zipResp.status()).toBe(200);
    expect(zipResp.headers()['content-type']).toContain('application/zip');
    expect(dl2.suggestedFilename()).toBe('batch_designs.zip');
  });
});

// ===========================================================================
// model-selector
// ===========================================================================
test.describe('model-selector', () => {
  test('Model dropdown renders with all three options and defaults to Opus', async ({
    freshPage: page,
  }) => {
    const select = page.locator('#model-select');
    await expect(select).toBeVisible();

    const options = select.locator('option');
    await expect(options).toHaveCount(3);

    await expect(options.nth(0)).toHaveAttribute('value', 'claude-opus-4-6');
    await expect(options.nth(1)).toHaveAttribute('value', 'claude-sonnet-4-6');
    await expect(options.nth(2)).toHaveAttribute(
      'value',
      'claude-haiku-4-5-20251001',
    );

    await expect(options.nth(0)).toHaveText('Opus 4.6');
    await expect(options.nth(1)).toHaveText('Sonnet 4.6');
    await expect(options.nth(2)).toHaveText('Haiku 4.5');

    await expect(select).toHaveValue('claude-opus-4-6');
  });

  test('Switching model updates the selected value for each option', async ({
    freshPage: page,
  }) => {
    const select = page.locator('#model-select');

    await select.selectOption('claude-sonnet-4-6');
    await expect(select).toHaveValue('claude-sonnet-4-6');

    await select.selectOption('claude-haiku-4-5-20251001');
    await expect(select).toHaveValue('claude-haiku-4-5-20251001');

    await select.selectOption('claude-opus-4-6');
    await expect(select).toHaveValue('claude-opus-4-6');

    await expect(select).toBeEnabled();
  });

  test('Model selection persists after clicking New Chat', async ({
    freshPage: page,
  }) => {
    const select = page.locator('#model-select');

    await select.selectOption('claude-sonnet-4-6');
    await page.locator('.new-chat-btn').click();
    await expect(page.locator('#welcome')).toBeVisible();
    await expect(select).toHaveValue('claude-sonnet-4-6');

    await select.selectOption('claude-haiku-4-5-20251001');
    await page.locator('.new-chat-btn').click();
    await expect(select).toHaveValue('claude-haiku-4-5-20251001');
  });

  test('Selected model is included in /api/chat request payload', async ({
    freshPage: page,
  }) => {
    const captured: any[] = [];
    await page.route('**/api/chat', async (route) => {
      captured.push({
        body: route.request().postDataJSON(),
        headers: route.request().headers(),
      });
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: MOCK_SSE_DONE,
      });
    });

    // First: Haiku
    await page.locator('#model-select').selectOption('claude-haiku-4-5-20251001');
    await page.locator('#input').fill('test message');
    await page.locator('#send-btn').click();
    await expect(page.locator('#send-btn')).toBeVisible({ timeout: 5_000 });

    expect(captured).toHaveLength(1);
    expect(captured[0].body.model).toBe('claude-haiku-4-5-20251001');
    expect(captured[0].body.message).toBe('test message');
    expect(captured[0].headers['content-type']).toContain('application/json');

    // Repeat with Sonnet
    await page.locator('#model-select').selectOption('claude-sonnet-4-6');
    await page.locator('#input').fill('second message');
    await page.locator('#send-btn').click();
    await expect(page.locator('#send-btn')).toBeVisible({ timeout: 5_000 });

    expect(captured).toHaveLength(2);
    expect(captured[1].body.model).toBe('claude-sonnet-4-6');
  });

  test('Model selection persists when switching between existing sessions', async ({
    page,
    waitForHealthy,
  }) => {
    await page.addInitScript(() => sessionStorage.clear());

    await page.route('**/api/sessions', (route) => {
      if (route.request().method() !== 'GET') return route.continue();
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            session_id: 'sess-a',
            first_message: 'Session A',
            created_at: '2026-01-01T00:00:00Z',
            outcomes_count: 0,
          },
          {
            session_id: 'sess-b',
            first_message: 'Session B',
            created_at: '2026-01-02T00:00:00Z',
            outcomes_count: 0,
          },
        ]),
      });
    });
    await page.route('**/api/sessions/*/messages', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      }),
    );

    await page.goto('/');
    await waitForHealthy(page);
    await expect(page.locator('#sessions-list .session-item')).toHaveCount(2);

    const select = page.locator('#model-select');
    await select.selectOption('claude-sonnet-4-6');

    await page.locator('.session-item').filter({ hasText: 'Session A' }).click();
    await expect(select).toHaveValue('claude-sonnet-4-6');

    await page.locator('.session-item').filter({ hasText: 'Session B' }).click();
    await expect(select).toHaveValue('claude-sonnet-4-6');

    const storedId = await page.evaluate(() =>
      sessionStorage.getItem('plasmid_session_id'),
    );
    expect(storedId).toBe('sess-b');
  });
});

// ===========================================================================
// error-handling
// ===========================================================================
test.describe('error-handling', () => {
  test('Missing API key shows graceful error in chat stream', async ({
    freshPage: page,
  }) => {
    // Simulate the server-side auth failure via route interception so we don't
    // have to restart the webServer with an unset key.
    await page.route('**/api/chat', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body:
          'data: {"type":"error","message":"Invalid or missing ANTHROPIC_API_KEY."}\n\n' +
          'data: {"type":"done"}\n\n',
      }),
    );

    const consoleErrors: string[] = [];
    page.on('pageerror', (e) => consoleErrors.push(e.message));

    await page.locator('#input').fill('Design an EGFP expression plasmid');
    await page.locator('#send-btn').click();

    await expect(page.locator('.msg.assistant')).toContainText(
      /Error: Invalid or missing ANTHROPIC_API_KEY\./,
    );
    await expect(page.locator('#send-btn')).toBeVisible();
    await expect(page.locator('#stop-btn')).toBeHidden();
    await expect(page.locator('#input')).toBeEnabled();
    await expect(page.locator('#welcome')).toBeHidden();
    expect(consoleErrors).toHaveLength(0);
  });

  test('Empty/whitespace input is silently rejected client-side', async ({
    freshPage: page,
  }) => {
    const chatCalls: string[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/chat')) chatCalls.push(req.url());
    });

    await expect(page.locator('#welcome')).toBeVisible();
    await page.locator('#input').fill('   ');
    await page.locator('#input').press('Enter');
    await page.waitForTimeout(500);

    expect(chatCalls).toHaveLength(0);
    await expect(page.locator('#welcome')).toBeVisible();
    await expect(page.locator('#messages .msg.user')).toHaveCount(0);
    await expect(page.locator('#input')).toBeEnabled();
    await expect(page.locator('#send-btn')).toBeVisible();
    await expect(page.locator('#stop-btn')).toBeHidden();
  });

  test('Health badge transitions offline on network failure and recovers', async ({
    freshPage: page,
  }) => {
    const consoleErrors: string[] = [];
    page.on('pageerror', (e) => consoleErrors.push(e.message));

    await expect(page.locator('#health-text')).toContainText('Agent Online');
    await expect(page.locator('#health-badge')).toHaveClass(/online/);

    await page.route('**/api/health', (route) => route.abort());
    // Trigger an immediate poll rather than waiting the full 5s interval
    await page.evaluate(() => (window as any).checkHealth?.());

    await expect(page.locator('#health-badge')).toHaveClass(/offline/, {
      timeout: 6_000,
    });
    await expect(page.locator('#health-badge')).not.toHaveClass(/online/);
    await expect(page.locator('#health-text')).toHaveText('Agent Offline');

    await page.unroute('**/api/health');
    await page.evaluate(() => (window as any).checkHealth?.());

    await expect(page.locator('#health-badge')).toHaveClass(/online/, {
      timeout: 6_000,
    });
    await expect(page.locator('#health-text')).toHaveText('Agent Online');
    expect(consoleErrors).toHaveLength(0);
  });

  test('Chat stream network failure shows connection error and restores UI', async ({
    freshPage: page,
  }) => {
    await page.route('**/api/chat', (route) => route.abort('failed'));

    await page.locator('#input').fill('What backbones are available?');
    await page.locator('#send-btn').click();

    await expect(page.locator('.msg.assistant')).toContainText(
      /Connection error:/,
      { timeout: 10_000 },
    );
    await expect(page.locator('.msg.user')).toContainText(
      'What backbones are available?',
    );
    await expect(page.locator('#send-btn')).toBeVisible();
    await expect(page.locator('#stop-btn')).toBeHidden();
    await expect(page.locator('#input')).toBeEnabled();
    await expect(page.locator('#input')).toBeFocused();
    await expect(page.locator('.streaming-cursor')).toHaveCount(0);
  });

  test('Stale session_id returns 404 and does not silently create new session', async ({
    page,
    request,
  }) => {
    const staleId = 'nonexistent-session-abc123';
    const resp = await request.post('/api/chat', {
      data: { message: 'hello', session_id: staleId },
    });

    expect(resp.status()).toBe(404);
    expect(resp.headers()['content-type']).toContain('application/json');
    const body = await resp.json();
    expect(body.error).toMatch(
      /Session not found.*expired or been cleared.*start a new conversation/i,
    );

    // Confirm server did not silently create the session
    const listResp = await request.get('/api/sessions');
    const sessions = await listResp.json();
    expect(
      sessions.find((s: any) => s.session_id === staleId),
    ).toBeUndefined();

    // Mark page as used to satisfy the fixture contract
    void page;
  });
});
