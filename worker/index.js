/**
 * Webhook relay.
 *
 * Holds no media and no state: it validates an inbound request, then asks CI to
 * do the real work. Anything that fails validation is answered with a plain 200
 * so that the caller learns nothing.
 */

const MAX_BYTES = 19 * 1024 * 1024;

// The uploader's endpoint. A secret path segment rather than a header, because a
// watch app is unlikely to support custom headers, and it is the pattern Telegram
// itself recommends for webhooks.
const UPLOAD_PATH = "/w/";

const ok = () => new Response("OK", { status: 200 });

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Two endpoints with opposite contracts, kept deliberately apart. This one is
    // for the recorder and must tell the caller the truth, because the caller marks
    // a recording as uploaded on success - a comforting 200 would make it mark a
    // file synced that never arrived. The webhook below is the opposite: it answers
    // 200 to everything so a prober learns nothing.
    if (url.pathname.startsWith(UPLOAD_PATH)) return upload(request, env, ctx, url);

    if (request.method !== "POST") return ok();

    const presented = request.headers.get("X-Telegram-Bot-Api-Secret-Token") || "";
    if (!safeEqual(presented, env.TELEGRAM_WEBHOOK_SECRET || "")) return ok();

    let update;
    try {
      update = await request.json();
    } catch {
      return ok();
    }

    // Every update type that can carry a message. `channel_post` matters: a
    // recorder that posts into a channel produces a channel_post, not a message,
    // and a relay that looks only at `message` drops it with no error anywhere -
    // which is the same silent failure the watch uploads showed.
    const message =
      update.message ||
      update.edited_message ||
      update.channel_post ||
      update.edited_channel_post ||
      update.business_message ||
      update.edited_business_message;
    if (!message || !message.chat) return ok();

    // TELEGRAM_CHAT_ID may hold more than one id, comma separated, so a second
    // source (a channel the recorder posts into, say) can be allowed later
    // without a redeploy.
    const allowed = String(env.TELEGRAM_CHAT_ID || "")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);

    if (!allowed.includes(String(message.chat.id))) {
      // A message from somewhere other than the owner's chat. Never reply to the
      // sender - that would confirm the bot is alive - but do tell the owner, in
      // the owner's own chat. This is how a file sent from a different account or
      // a different chat becomes visible instead of vanishing, which is exactly
      // why the watch uploads looked like they had never been sent at all. The id
      // is reported so it can be pasted straight into TELEGRAM_CHAT_ID.
      const shape = describeUpload(message);
      const owner = allowed[0];
      if (shape && owner) {
        ctx.waitUntil(say(env, owner, `Message from another chat (id ${message.chat.id}): ${shape}`));
      }
      return ok();
    }

    // A one-tap liveness check. Silent rejection is the whole problem this relay
    // has: without a command that always answers, "the relay is down" and "the file
    // was never delivered to it" look identical from inside the chat. Send /ping
    // and a reply proves the webhook, the worker and the bot token all work.
    if (typeof message.text === "string" && message.text.trim() === "/ping") {
      ctx.waitUntil(say(env, message.chat.id, "pong"));
      return ok();
    }

    const picked = pickMedia(message);
    if (!picked) {
      // An upload we did not recognise is reported back to the chat. This is the
      // only way to see it: the worker answers 200 either way, so Telegram records
      // no error and pending_update_count stays 0 - which makes a silent rejection
      // look exactly like a file that never arrived.
      //
      // Only uploads are reported; plain text, photos and stickers pass without
      // comment so the chat stays quiet. The chat id was checked above, so this
      // goes to the owner and nobody else.
      if (message.document || message.video || message.video_note) {
        ctx.waitUntil(say(env, message.chat.id, `Ignored an upload: ${describeUpload(message)}`));
      }
      return ok();
    }

    const { media, kind, fileName } = picked;
    if (!media.file_id) return ok();

    if (media.file_size && media.file_size > MAX_BYTES) {
      ctx.waitUntil(say(env, message.chat.id, "That file is too large."));
      return ok();
    }

    ctx.waitUntil(dispatch(env, media.file_id, message.chat.id, message.message_id, kind, fileName));
    return ok();
  },
};

/**
 * A short, non-sensitive description of an upload, for diagnosing a rejection.
 * Returns "" when there is nothing to describe, so callers can use it as the gate:
 * a message with no media must not produce a notification.
 */
function describeUpload(message) {
  const parts = [];
  for (const field of ["document", "video", "video_note", "audio", "voice"]) {
    if (message[field]) parts.push(field);
  }
  const media = message.document || message.video || message.video_note;
  if (media) {
    if (typeof media.mime_type === "string") parts.push(`mime=${media.mime_type}`);
    if (typeof media.file_name === "string") parts.push(`name=${media.file_name}`);
    if (typeof media.file_size === "number") parts.push(`size=${media.file_size}`);
  }
  return parts.join(" ");
}

/**
 * Audio only. Returns the media object plus what kind of upload it was, or null.
 *
 * An automated recorder (a watch, a phone) uploads .m4a as a *document*, and the
 * mime type it declares is not dependable - some senders use
 * application/octet-stream, which would fail a mime-only check and drop the file
 * with no error anywhere. So accept an audio mime type OR a known audio filename
 * extension. The list covers what recorders actually produce, including the .3gp
 * and .amr that older handsets and watches still emit.
 */
const AUDIO_EXTENSIONS = [
  ".m4a", ".mp3", ".wav", ".ogg", ".oga", ".opus", ".aac", ".flac",
  ".3gp", ".amr", ".weba", ".aiff", ".aif", ".caf", ".wma", ".m4b",
];

/**
 * The uploader's own filename, or "" when it did not send one.
 *
 * Only ever used to name the cleaned file that is sent back, and never logged.
 * Kept in its original case - the lowercased copy inside pickMedia exists only
 * so that the extension test is case-insensitive.
 */
function originalName(media) {
  return typeof media.file_name === "string" ? media.file_name : "";
}

function pickMedia(message) {
  if (message.voice) return { media: message.voice, kind: "voice", fileName: "" };
  if (message.audio) return { media: message.audio, kind: "audio", fileName: originalName(message.audio) };
  const doc = message.document;
  if (!doc) return null;
  const mime = typeof doc.mime_type === "string" ? doc.mime_type.toLowerCase() : "";
  const name = typeof doc.file_name === "string" ? doc.file_name.toLowerCase() : "";
  if (mime.startsWith("audio/") || AUDIO_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return { media: doc, kind: "document", fileName: originalName(doc) };
  }
  return null;
}

/** Constant-time comparison so the secret token cannot be probed byte by byte. */
function safeEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function json(body, status) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * The recorder's door.
 *
 * The recorder used to call Telegram itself, which is exactly why its recordings
 * were invisible: a bot's own message is never delivered back to it, so the relay
 * was never told a file existed. Now it posts here instead, and this function is
 * the one that talks to Telegram - so it learns the file_id from the reply and can
 * hand the work to CI.
 *
 * The caller's multipart body is forwarded verbatim, so the boundary and the field
 * names survive untouched and the file is never buffered: a ten minute m4a is about
 * 10 MB and there is no reason to hold it in memory.
 */
async function upload(request, env, ctx, url) {
  if (request.method !== "POST") return json({ ok: false, description: "method not allowed" }, 405);

  const presented = url.pathname.slice(UPLOAD_PATH.length).replace(/\/+$/, "");
  if (!safeEqual(presented, env.UPLOAD_SECRET || "")) {
    // An unset UPLOAD_SECRET closes the door entirely, and a wrong one says nothing
    // that would confirm the endpoint exists.
    return json({ ok: false, description: "not found" }, 404);
  }

  let response;
  try {
    response = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendDocument`, {
      method: "POST",
      headers: { "Content-Type": request.headers.get("Content-Type") || "" },
      body: request.body,
    });
  } catch {
    return json({ ok: false, description: "could not reach Telegram" }, 502);
  }

  let text = "";
  try {
    text = await response.text();
  } catch {
    return json({ ok: false, description: "unreadable reply from Telegram" }, 502);
  }

  let fileId = "";
  let chatId = env.TELEGRAM_CHAT_ID;
  let messageId = null;
  let fileName = "";
  try {
    const message = JSON.parse(text).result || {};
    const media = message.document || message.audio || message.voice || {};
    fileId = media.file_id || "";
    fileName = originalName(media);
    if (message.chat && message.chat.id) chatId = message.chat.id;
    if (message.message_id) messageId = message.message_id;
  } catch {
    // Not JSON. It is still returned as-is, so the caller sees whatever Telegram said.
  }

  // The recording is safely inside Telegram by now, so a failed dispatch must NOT
  // make the caller think the upload failed - it would send again and the file
  // would arrive twice. Report the dispatch problem to the owner instead.
  if (fileId) ctx.waitUntil(dispatch(env, fileId, chatId, messageId, "document", fileName));

  // Telegram's own status and body, so the caller's existing success check keeps
  // working without changing.
  return new Response(text, { status: response.status, headers: { "Content-Type": "application/json" } });
}

/**
 * The workflow's concurrency group lets 100 runs wait their turn. Past that,
 * GitHub still answers a dispatch with 204 - it accepts the request, creates the
 * run, and then fails it immediately with *no jobs at all*. A run with no job
 * never executes the `notify` step either, so the file is dropped without one
 * message anywhere. Measured over the last 200 runs, 23 failed exactly that way:
 * every failure in the window, and roughly a fifth of the real work.
 *
 * That is invisible from the dispatch response, which is why this exists: ask how
 * deep the queue already is before adding to it. The limit sits below the real cap
 * because several uploads can be in flight at once and each of them reads the same
 * count.
 */
const QUEUE_LIMIT = 90;

/** The headers every GitHub call needs. One definition, so they cannot drift. */
function githubHeaders(env) {
  return {
    Authorization: `Bearer ${env.GITHUB_DISPATCH_TOKEN}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "media-utils-lab",
  };
}

/**
 * How many runs are waiting to start, or null when that cannot be read.
 *
 * Both waiting states are counted, because either one alone undercounts: a run
 * held back by the concurrency group reports `pending`, while a run waiting for a
 * free runner reports `queued`.
 *
 * null means "could not tell", and the caller treats that as permission to go
 * ahead. A guard that failed closed would turn a GitHub hiccup into a relay that
 * refuses every upload - a worse outage than the one it prevents.
 */
async function queueDepth(env) {
  let total = 0;
  for (const status of ["pending", "queued"]) {
    let response;
    try {
      response = await fetch(
        `https://api.github.com/repos/${env.GITHUB_REPO}/actions/runs?status=${status}&per_page=1`,
        { headers: githubHeaders(env) },
      );
    } catch {
      return null;
    }
    if (response.status !== 200) return null;
    let body;
    try {
      body = await response.json();
    } catch {
      return null;
    }
    if (typeof body.total_count !== "number") return null;
    total += body.total_count;
  }
  return total;
}

/**
 * True at most once per window, so a burst of refused uploads produces one message
 * instead of one per file - at a hundred files an hour, a message each is the same
 * noise problem the routine notices had.
 *
 * Best effort by design: the cache is per-colo and may miss, and a missed window
 * only ever means one extra message. It can never cause a silent drop, which is
 * the failure this whole guard exists to remove.
 */
async function firstInWindow(key, seconds) {
  try {
    const cache = caches.default;
    const url = `https://queue-guard.invalid/${key}`;
    if (await cache.match(url)) return false;
    await cache.put(
      url,
      new Response("1", { headers: { "Cache-Control": `max-age=${seconds}` } }),
    );
    return true;
  } catch {
    // No cache available. Say it every time rather than not at all.
    return true;
  }
}

async function dispatch(env, fileId, chatId, messageId, kind, fileName = "") {
  // Refusing here is not a silent drop - it is a file the owner is told about and
  // can send again, which is the whole difference this makes. Dispatching into a
  // full queue would lose it without a word.
  const depth = await queueDepth(env);
  if (depth !== null && depth >= QUEUE_LIMIT) {
    if (await firstInWindow("queue-full", 300)) {
      await say(
        env,
        chatId,
        `The job queue is full (${depth} waiting), so uploads are not being queued. Send them again once it drains.`,
      );
    }
    return;
  }

  try {
    const response = await fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/dispatches`, {
      method: "POST",
      headers: {
        ...githubHeaders(env),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        event_type: "process_audio",
        // `kind` is a fixed word, never user data: an automated upload arrives as
        // a document and gets a transcript plus the cleaned audio, a hand-sent
        // voice note also gets the cleaned audio back.
        //
        // `file_name` is the uploader's own filename, carried so that the cleaned
        // file can come back suffixed with it. Like the ids above it is per-file
        // data and must never be logged - which is why the workflow reads this
        // payload from the event file instead of echoing it into the run log.
        client_payload: {
          file_id: fileId,
          chat_id: chatId,
          message_id: messageId,
          kind,
          file_name: fileName || "",
        },
      }),
    });
    // 204 is the documented success status for a dispatch.
    if (response.status !== 204) {
      // The status alone is the diagnosis, and none of it is sensitive: 401 means
      // the token is rejected or expired, 403 means it cannot reach the repo, 404
      // means the repo name is wrong or invisible to the token, 422 means the event
      // type is not one the workflow listens for. Reporting it turns the last
      // silent failure - "the file vanished and nothing anywhere said why" - into
      // something actionable.
      await say(env, chatId, `Could not queue the job (http ${response.status}).`);
    } else {
      await say(env, chatId, "Queued.");
    }
  } catch {
    // A thrown fetch error carries the failed request, and that request carries the
    // token in its Authorization header, so the error text is never echoed. The
    // failure itself must still be visible: otherwise a network fault looks exactly
    // like a file that was never sent.
    await say(env, chatId, "Could not reach GitHub to queue the job.");
  }
}

async function say(env, chatId, text) {
  try {
    await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chatId, text }),
    });
  } catch {
    // Nothing useful to do here.
  }
}
