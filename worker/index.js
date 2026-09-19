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

    const message = update.message || update.edited_message;
    if (!message || !message.chat) return ok();
    if (String(message.chat.id) !== String(env.TELEGRAM_CHAT_ID)) return ok();

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

/** A short, non-sensitive description of an upload, for diagnosing a rejection. */
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
  return parts.join(" ") || "unknown";
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
      await say(env, chatId, "Could not queue the job.");
    } else {
      await say(env, chatId, "Queued.");
    }
  } catch {
    // Stay silent: thrown fetch errors can carry the token in their message.
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
