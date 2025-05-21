import { extractFromHtml } from "@extractus/article-extractor";

async function parseArticle() {
  process.stdin.setEncoding('utf8');
  process.stdout.setEncoding('utf8');

  let html = '';

  process.stdin.on('data', chunk => {
    html += chunk;
  });

  process.stdin.on('end', async () => {
    try{
      const article = await extractFromHtml(html)
      console.log(JSON.stringify(article));
    }
    catch {
      console.log('[]')
    }
  });
}

// Run the extraction
parseArticle()

