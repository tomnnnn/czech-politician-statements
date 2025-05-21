import { extract } from "@extractus/article-extractor";
import { URL } from "url";
import cliProgress from "cli-progress";
import {HeaderGenerator} from 'header-generator';
import fs from 'fs';
import { RateLimiter } from "limiter";
import Database from "better-sqlite3";
import SqliteError from "better-sqlite3"
import pLimit from 'p-limit';

let headerGenerator = new HeaderGenerator({
        browsers: [
            {name: "firefox", minVersion: 80},
            {name: "chrome", minVersion: 87},
            "safari"
        ],
        devices: [
            "desktop"
        ],
        operatingSystems: [
            "windows"
        ]
});

// Limit settings
const REQUESTS_PER_DOMAIN_PER_SECOND = 2;
const DOMAIN_LIMITERS = new Map();

// Initialize SQLite database
const db = new Database("datasets/demagog.db");

db.exec(`
  CREATE TABLE IF NOT EXISTS failed_scrapes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_id INTEGER,
    url TEXT,
    error TEXT,
    timestamp TEXT,
    evidence_source TEXT,
    FOREIGN KEY (statement_id) REFERENCES statements(id)
  );
`);

function logFailedScrape(statement_id, url, error) {
  const stmt = db.prepare(`
      INSERT INTO failed_scrapes (statement_id, url, error, timestamp)
      VALUES (?, ?, ?, ?)
  `);
  stmt.run(statement_id, url, error.message, new Date().toISOString());
}

function linkArticleWithStatement(statement_id, article_id) {
  console.log(`Linking statement ${statement_id} with article ${article_id}`);
  const stmt = db.prepare(`
      INSERT OR IGNORE INTO article_relevance (statement_id, article_id)
      VALUES (?, ?)
  `);
  stmt.run(statement_id, article_id);
}

// Function to save article data to SQLite
function saveArticle(statement_ids, article) {
  console.log(`Saving article: ${article.url}`);

  const stmt = db.prepare(`
      INSERT OR IGNORE INTO articles (url, title, description, content, type, author, source, published, accessed)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run(
    article.url,
    article.title,
    article.description,
    article.content,
    article.type,
    article.author,
    article.source,
    article.published,
    new Date().toISOString(),
  );

  console.log(`Article saved: ${article.url}`);

  // Try to get the article ID
  let articleId;
  const lastId = db.prepare("SELECT last_insert_rowid() as id").get().id;

  if (lastId !== 0) {
    console.log(`Article ID: ${lastId}`);
    articleId = lastId;
  } else {
    console.log('Article already exists, trying to get its ID');
    articleId = getArticleId(article.url);
  }

  // Link the article with the statement
  if (articleId) {
    statement_ids.forEach(statement_id => {
      linkArticleWithStatement(statement_id, articleId);
    });
  }
}
// Create progress bar
const progressBar = new cliProgress.SingleBar(
  {
    format:
      "Processing [{bar}] {percentage}% | {value}/{total} URLs | ETA: {eta}s",
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
    DOMAIN_LIMITERS.set(domain, new RateLimiter({
      tokensPerInterval: REQUESTS_PER_DOMAIN_PER_SECOND,
      interval: "second"
    }));
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

function getArticleId(url) {
  // Check if the article already exists in the database. If yes, return its ID.
  const stmt = db.prepare("SELECT id FROM articles WHERE url = ?");

  const row = stmt.get(url);
  if (row) {
    return row.id;
  }
  return null;
}

// Processing each URL with rate limiting
async function processUrl(statement_ids, url) {
  if (!isValidUrl(url)) {
    console.warn(`Skipping invalid URL: ${url}`);
    return;
  }
  // Check if the URL is already in the database
  const existingArticleId = getArticleId(url);

  if (existingArticleId) {
    // try to link the article with statements
    statement_ids.forEach(statement_id => {
      linkArticleWithStatement(statement_id, existingArticleId);
    });

    console.warn(`Article already exists in the database: ${url}`);
    return
  }

  const domain = new URL(url).hostname;
  const limiter = getLimiterForDomain(domain);
  await limiter.removeTokens(1); // Rate limit for the domain

  let headers = headerGenerator.getHeaders({
      operatingSystems: [
          "windows"
      ],
      locales: ["en-US", "en", "cs-CZ", "cs"],
  });

  try {
    const article = await extract(url,{
      signal: AbortSignal.timeout(5000)
    },{
      headers: headers, 
    });

    saveArticle(statement_ids, article);
  } catch (error) {
    if (error instanceof SqliteError) {
      throw error;
    }
    statement_ids.forEach(statement_id => {
      logFailedScrape(statement_id, url, error);
    })
  }
  finally {
      progressBar.increment(); // Update progress bar
  }
}

async function extractArticles() {
  // Get the file path from the argument
  const args = process.argv.slice(2);
  if (args.length !== 1) {
    console.error('Please provide the path to the JSON file.');
    process.exit(1);
  }

  const filePath = args[0];

  // expecting format {
  //  url: [statement_id_1, statement_id_2, ...]
  // }
  
  let urls;
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    urls = JSON.parse(fileContent);
  } catch (error) {
    console.error(`Error reading or parsing file: ${error.message}`);
    process.exit(1);
  }

  const total_urls = Object.keys(urls).length;
  progressBar.start(total_urls, 0);

  const limit = pLimit(100);

  const promises = Object.entries(urls).map(([url, statement_ids]) => 
    limit(() => processUrl(statement_ids, url))
  );

  await Promise.all(promises);

  progressBar.stop();
}

// Run the extraction
extractArticles();

