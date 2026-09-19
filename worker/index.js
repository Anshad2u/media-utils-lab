/**
 * Webhook relay.
 *
 * Holds no media and no state: it validates an inbound request, then asks CI to
 * do the real work. Anything that fails validation is answered with a plain 200
 * so that the caller learns nothing.
 */

const MAX_BYTES = 19 * 1024 * 1024;

const ok = () => new Response("OK", { status: 200 });

export default {
  async fetch(request, env, ctx) {
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

    const { media, kind } = picked;
    if (!media.file_id) return ok();

    if (media.file_size && media.file_size > MAX_BYTES) {
      ctx.waitUntil(say(env, message.chat.id, "That file is too large."));
      return ok();
    }

    ctx.waitUntil(dispatch(env, media.file_id, message.chat.id, message.message_id, kind));
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

function pickMedia(message) {
  if (message.voice) return { media: message.voice, kind: "voice" };
  if (message.audio) return { media: message.audio, kind: "audio" };
  const doc = message.document;
  if (!doc) return null;
  const mime = typeof doc.mime_type === "string" ? doc.mime_type.toLowerCase() : "";
  const name = typeof doc.file_name === "string" ? doc.file_name.toLowerCase() : "";
  if (mime.startsWith("audio/") || AUDIO_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return { media: doc, kind: "document" };
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

async function dispatch(env, fileId, chatId, messageId, kind) {
  try {
    const response = await fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/dispatches`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_DISPATCH_TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "media-utils-lab",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        event_type: "process_audio",
        // `kind` is a fixed word, never user data: an automated upload arrives as
        // a document and gets a transcript only, a hand-sent voice note also gets
        // the cleaned audio back.
        client_payload: { file_id: fileId, chat_id: chatId, message_id: messageId, kind },
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
