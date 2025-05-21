import { extract } from "@extractus/article-extractor";
import { setTimeout } from "timers/promises";
import pLimit from "p-limit";
import { URL } from "url";
import cliProgress from "cli-progress";
import UserAgent from "user-agents";
import { readFileSync } from 'fs';

// Limit settings
const REQUESTS_PER_DOMAIN_PER_SECOND = 3;
const DOMAIN_LIMITERS = new Map();

// Create progress bar
const progressBar = new cliProgress.SingleBar(
  {
    format:
      "Processing [{bar}] {percentage}% | {value}/{total} URLs | ETA: {eta}s | Speed: {speed} URLs/s",
    barCompleteChar: "=",
    barIncompleteChar: " ",
    hideCursor: false,
    stream: process.stderr,
  },
  cliProgress.Presets.shades_classic,
);

// Function to create a limiter for each domain
function getLimiterForDomain(domain) {
  if (!DOMAIN_LIMITERS.has(domain)) {
    DOMAIN_LIMITERS.set(domain, pLimit(REQUESTS_PER_DOMAIN_PER_SECOND));
  }
  return DOMAIN_LIMITERS.get(domain);
}

// Function to check if a URL is valid
function isValidUrl(url) {
  try {
    new URL(url);
    return true;
  } catch (error) {
    return false;
  }
}

// Processing each URL with rate limiting
async function processUrl(url) {
  if (!isValidUrl(url)) {
    console.warn(`Skipping invalid URL: ${url}`);
    return;
  }

  const domain = new URL(url).hostname;
  const limiter = getLimiterForDomain(domain);

  return limiter(async () => {
    await setTimeout(5000 / REQUESTS_PER_DOMAIN_PER_SECOND); // Rate limit delay

    try {
      const article = await extract(url, {
        headers: {
          "user-agent": new UserAgent().toString(),
        },
      });

      return {
        success: true,
        data: article,
      }
    } catch (error) {
      return {
        success: false,
        data: {
          url: url,
          error: error.message,
        }
      }
    }
    finally {
        progressBar.increment(); // Update progress bar
    }

  });
}

async function extractArticles() {
  // load urls from --urls argument (passes json string)
  var urls = [];


const args = process.argv.slice(2);
if (args.length === 1) {
const filePath = args[0];
const fileContent = readFileSync(filePath, 'utf-8');
urls = JSON.parse(fileContent);
}

  progressBar.start(urls.length, 0);

  const promises = urls.map( url =>
    processUrl(url),
  );
  
  const result = await Promise.all(promises);

  const articles = result
    .filter(item => item.success)
    .map(item => item.data)

  const unprocessed = result
    .filter(item => !item.success)
    .map(item => item.data)

  const result_obj = {
    "articles": articles,
    "unprocessed": unprocessed,
  }
  const result_json = JSON.stringify(result_obj, null, 2);

  progressBar.stop();

  console.log(result_json);

}

// Run the extraction
extractArticles()

