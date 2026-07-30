const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const sourcePath = path.join(root, 'doca.py');
const outPath = path.join(root, 'Monitoringsnotitie_AI_en_Werk_bijgewerkt.md');
const source = fs.readFileSync(sourcePath, 'utf8');

function parenBalance(text) {
  let balance = 0;
  let quote = null;
  let escaped = false;

  for (const ch of text) {
    if (quote) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === quote) {
        quote = null;
      }
      continue;
    }

    if (ch === "'" || ch === '"') {
      quote = ch;
    } else if (ch === '(') {
      balance += 1;
    } else if (ch === ')') {
      balance -= 1;
    }
  }

  return balance;
}

function pyStringLiterals(block) {
  const matches = [];
  const regex = /'((?:\\.|[^'\\])*)'/g;
  let match;

  while ((match = regex.exec(block)) !== null) {
    matches.push(Function(`"use strict"; return '${match[1]}';`)());
  }

  return matches;
}

function collectCalls() {
  const lines = source.split(/\r?\n/);
  const calls = [];
  const prefixes = ['bold_para(doc', 'norm_para(doc', 'italic_para(doc', 'bibsub(doc', 'bibentry(doc', 'blank(doc)'];

  for (let i = 0; i < lines.length; i += 1) {
    const trimmed = lines[i].trimStart();
    const prefix = prefixes.find((candidate) => trimmed.startsWith(candidate));
    if (!prefix) continue;

    let block = lines[i];
    let balance = parenBalance(block);
    while (balance > 0 && i + 1 < lines.length) {
      i += 1;
      block += `\n${lines[i]}`;
      balance += parenBalance(lines[i]);
    }
    calls.push(block);
  }

  return calls;
}

function renderCall(block, state) {
  const strings = pyStringLiterals(block);
  if (block.trimStart().startsWith('blank(doc)')) return '';

  if (block.includes('bold_para(doc')) {
    const title = strings.join('');
    if (!state.seenTitle) {
      state.seenTitle = true;
      return `# ${title}`;
    }
    return `## ${title}`;
  }

  if (block.includes('bibsub(doc')) {
    return `### ${strings.join('')}`;
  }

  if (block.includes('italic_para(doc')) {
    return `*${strings.join('')}*`;
  }

  if (block.includes('bibentry(doc')) {
    const [title, url, ...noteParts] = strings;
    const note = noteParts.join('');
    return note
      ? `**${title}.** ${url}\n\n*${note}*`
      : `**${title}.** ${url}`;
  }

  if (block.includes('norm_para(doc')) {
    return strings.join('');
  }

  return '';
}

const state = { seenTitle: false };
const markdown = collectCalls()
  .map((block) => renderCall(block, state))
  .filter((text) => text.length > 0)
  .join('\n\n');

fs.writeFileSync(outPath, `${markdown}\n`, 'utf8');
console.log(outPath);
