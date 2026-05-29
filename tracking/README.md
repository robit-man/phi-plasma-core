# Tracked Social Preview Image

Self-host the social preview image so you can see every link unfurl on
Twitter, LinkedIn, Slack, Discord, Telegram, Signal, etc. — even when
the recipient never clicks through.

## What this gives you

| Event | What you see |
|---|---|
| Someone tweets your repo URL | Twitterbot fetches the OG image → you get IP + UA + referrer + country |
| Someone posts the link in Slack/Discord | Slackbot/Discordbot fetches → you log it |
| LinkedIn share preview | LinkedInBot fetches → logged |
| Direct README view on GitHub | camo proxy fetches → logged (once per cache window) |
| OG-card preview on any platform with link unfurling | Their bot fetches → logged |

**Bots that don't matter** (Googlebot, Bingbot, Sourcegraph) are flagged
in the log so you can filter them in analysis. The signal you care about
is the platform-specific unfurl bots and the real humans behind them.

## Setup (~30 min one-time)

### Prerequisites

- A domain you control (you have `roko.network`)
- Cloudflare account (free tier is enough)
- `wrangler` CLI installed: `npm install -g wrangler`

### Steps

```bash
# 1. Authenticate
wrangler login

# 2. Create a KV namespace to store logs (90-day TTL)
wrangler kv:namespace create "TRACK_LOGS"
# Paste the returned `id` into wrangler.toml under [[kv_namespaces]]

# 3. (Optional) Add a Discord webhook for real-time alerts
#    Create webhook in your Discord server's channel settings,
#    then:
wrangler secret put LOG_WEBHOOK_URL
# Paste the webhook URL when prompted

# 4. Deploy
cd tracking/
wrangler deploy

# 5. Wire up the custom route
#    In wrangler.toml uncomment the [[routes]] block, set your domain
#    Example:  pattern = "img.roko.network/preview.png"
#              zone_name = "roko.network"
#    Then redeploy:
wrangler deploy

# 6. (DNS) Add an A record or CNAME pointing img.roko.network at
#    Cloudflare. If your DNS is already on Cloudflare, the Workers route
#    handles this automatically.

# 7. Update the GitHub repo's social preview + README to point at the new URL
```

### Update the repo

Once deployed, edit `docs/index.html` to use the tracked URL as the
`og:image`:

```html
<meta property="og:image" content="https://img.roko.network/preview.png" />
```

And add a tracked-image embed in `README.md` after the existing hero:

```markdown
<p align="center">
  <img src="https://img.roko.network/preview.png" alt="Φ-Plasma-Core" width="800"/>
</p>
```

## Reading the logs

```bash
# List all logged hits
wrangler kv:key list --namespace-id YOUR-KV-ID

# Dump a hit
wrangler kv:key get "hit:2026-05-29T12:34:56.789Z:abcd1234" --namespace-id YOUR-KV-ID

# Or query via the Cloudflare dashboard:
# https://dash.cloudflare.com → Workers & Pages → KV → TRACK_LOGS
```

For real-time visibility, point `LOG_WEBHOOK_URL` at a Discord/Slack/Telegram
webhook and you'll see each fetch in chat as it happens.

## What the logs look like

```json
{
  "ts": "2026-05-29T12:34:56.789Z",
  "ip": "104.244.42.7",
  "ua": "Twitterbot/1.0",
  "referer": "",
  "country": "US",
  "city": "San Francisco",
  "asn": 13414,
  "asOrg": "Twitter Inc.",
  "url": "https://img.roko.network/preview.png",
  "bot": true
}
```

The `bot: true` flag lets you filter aggregate-bot noise from
human-driven traffic. Bots fetching repeatedly are normal; the signal
is the *new* bots / new ASNs / new countries appearing over time.

## Cost

Cloudflare Workers free tier: 100,000 requests/day. KV: 100,000 reads
+ 1,000 writes/day. For a repo getting <1000 unfurls/day, this stays
at $0/mo indefinitely.

## Privacy note

This logs IP addresses of automated bots and any human who triggers an
OG-card fetch. It does NOT use cookies, fingerprinting, or any
session-based tracking. The data is purely incoming request metadata
that any web server collects by default.

If you care about GDPR compliance: the data collected (IP + UA +
country) qualifies as personal data under GDPR. Add a privacy notice to
your docs site if you're serving EU traffic and want to be strict. For
research-preview / personal-project scope, this is typically fine.
