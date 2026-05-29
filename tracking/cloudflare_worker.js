/**
 * Self-hosted social-preview image with full request logging.
 *
 * Why this exists
 * ---------------
 * GitHub serves images through a privacy-preserving proxy (camo.githubusercontent.com)
 * that caches images and obscures IP addresses. Hosting the social-preview
 * image yourself catches every link-unfurl on Twitter / Slack / Discord /
 * LinkedIn / Telegram / Signal — even when the user doesn't click.
 *
 * What gets logged (for every fetch)
 * ----------------------------------
 *   ts          — ISO timestamp
 *   ip          — CF-Connecting-IP (the real visitor IP, not Cloudflare's)
 *   ua          — User-Agent (identifies platform: Twitterbot, Slackbot, etc.)
 *   referer     — Referer header (where the image was embedded)
 *   country     — CF-IPCountry (ISO 2-letter code from Cloudflare edge)
 *   asn         — Autonomous System Number (which network / ISP)
 *   bot         — true if UA matches a known bot signature
 *
 * The logs go to a Cloudflare Workers KV namespace or, if you prefer,
 * stream to a webhook (Discord, Slack, your own server).
 *
 * Setup
 * -----
 * 1. wrangler login
 * 2. wrangler kv:namespace create "TRACK_LOGS"
 * 3. Set the namespace ID in wrangler.toml
 * 4. wrangler deploy
 * 5. Set the worker route to e.g. img.roko.network/preview.png
 * 6. Update GitHub repo Social Preview setting to point at this URL
 *    (Settings → General → Social preview → Upload image — but use the URL form
 *    by uploading the same file and noting GitHub still serves the cached
 *    bytes from their own CDN for the Social Preview slot itself. The
 *    tracking happens via README image embedding and via your OG image URL.)
 *
 * The README.md should embed:
 *   <img src="https://img.roko.network/preview.png" alt="Φ-Plasma-Core" />
 *
 * The docs/index.html should set:
 *   <meta property="og:image" content="https://img.roko.network/preview.png" />
 *
 * Key fact: Twitter/LinkedIn/Slack/Discord all fetch the OG image
 * server-side when an URL is shared. Each fetch hits this worker.
 * You see every unfurl event with full IP/UA/referer.
 */

const KNOWN_BOTS = [
  // Social platforms
  /Twitterbot/, /facebookexternalhit/, /LinkedInBot/, /Slackbot/,
  /Discordbot/, /TelegramBot/, /WhatsApp/, /SkypeUriPreview/,
  /Mastodon/, /Pleroma/, /BlueskyFetcher/,
  // Crawlers
  /Googlebot/, /bingbot/, /Yandexbot/, /Baiduspider/, /DuckDuckBot/,
  /Sourcegraph/, /Sogou/, /Applebot/,
  // Misc
  /curl/, /wget/, /Python-urllib/, /python-requests/, /httpx/, /Go-http-client/,
];

function isBot(ua) {
  if (!ua) return false;
  return KNOWN_BOTS.some(re => re.test(ua));
}

// Replace this with your image content (base64 or fetch from R2/KV).
// For simplicity here, we proxy from the GitHub-hosted asset.
const SOURCE_IMAGE_URL =
  "https://raw.githubusercontent.com/Prime-007-hash/phi-plasma-core/main/assets/social-preview-1280x640.png";

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Only handle the preview image route
    if (!url.pathname.endsWith("preview.png") && url.pathname !== "/") {
      return new Response("Not found", { status: 404 });
    }

    const cf = request.cf || {};
    const log = {
      ts: new Date().toISOString(),
      ip: request.headers.get("cf-connecting-ip") || "unknown",
      ua: request.headers.get("user-agent") || "",
      referer: request.headers.get("referer") || "",
      country: cf.country || "??",
      city: cf.city || "",
      asn: cf.asn || 0,
      asOrg: cf.asOrganization || "",
      url: request.url,
      bot: isBot(request.headers.get("user-agent")),
    };

    // Log via async write — doesn't block the response
    ctx.waitUntil(logHit(env, log));

    // Fetch and proxy the image. Cache aggressively to reduce origin load,
    // but bypass cache for the worker (we want each unfurl to hit us).
    const upstream = await fetch(SOURCE_IMAGE_URL, {
      cf: { cacheTtl: 3600, cacheEverything: true },
    });
    if (!upstream.ok) {
      return new Response("Upstream error", { status: 502 });
    }
    return new Response(upstream.body, {
      status: 200,
      headers: {
        "content-type": "image/png",
        // Don't let intermediate caches strip our visibility too aggressively
        "cache-control": "public, max-age=300, s-maxage=300",
        "access-control-allow-origin": "*",
      },
    });
  },
};

async function logHit(env, log) {
  // Strategy 1: KV namespace (cheap, queryable later)
  if (env.TRACK_LOGS) {
    const key = `hit:${log.ts}:${crypto.randomUUID().slice(0, 8)}`;
    await env.TRACK_LOGS.put(key, JSON.stringify(log), {
      expirationTtl: 60 * 60 * 24 * 90, // 90 days
    });
  }

  // Strategy 2: Webhook stream (optional — set DISCORD_WEBHOOK or your URL)
  if (env.LOG_WEBHOOK_URL) {
    try {
      await fetch(env.LOG_WEBHOOK_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          content:
            `**${log.bot ? "🤖" : "👤"} preview unfurl** — ` +
            `\`${log.country}\` ` +
            `\`${(log.ua || "").slice(0, 60)}\` ` +
            `from \`${(log.referer || "direct").slice(0, 60)}\``,
          embeds: [{ description: "```json\n" + JSON.stringify(log, null, 2).slice(0, 1500) + "\n```" }],
        }),
      });
    } catch (e) {
      // best-effort
    }
  }
}
