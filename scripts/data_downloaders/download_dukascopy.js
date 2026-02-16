const { getHistoricalRates } = require('dukascopy-node');
const fs = require('fs');
const path = require('path');

const TIMEFRAMES = { m1: 'm1', m5: 'm5', m15: 'm15', m30: 'm30', h1: 'h1', h4: 'h4', d1: 'd1' };

async function downloadData(instrument, timeframe = 'h1', days = 730, saveAs = null) {
    const filename = saveAs || instrument.toUpperCase();
    const endDate = new Date();
    const startDate = new Date(endDate - days * 24 * 60 * 60 * 1000);

    console.log(`\nTéléchargement: ${instrument} → ${filename}_${timeframe}.csv`);
    console.log(`Période: ${startDate.toISOString().split('T')[0]} → ${endDate.toISOString().split('T')[0]}`);

    const data = await getHistoricalRates({
        instrument: instrument.toLowerCase(),
        dates: { from: startDate, to: endDate },
        timeframe,
        format: 'json',
        batchSize: 30,
        pauseBetweenBatchesMs: 1000,
    });

    if (!data?.length) throw new Error('Aucune donnée reçue');

    const csv = 'timestamp,Open,High,Low,Close,Volume\n' +
        data.map(b => `${new Date(b.timestamp).toISOString()},${b.open},${b.high},${b.low},${b.close},${b.volume || 0}`).join('\n');

    const dataDir = path.join(__dirname, 'data');
    fs.mkdirSync(dataDir, { recursive: true });

    const filepath = path.join(dataDir, `${filename}_${timeframe}.csv`);
    fs.writeFileSync(filepath, csv);

    console.log(`✓ ${data.length} barres → ${filepath}\n`);
    return filepath;
}

// CLI
const args = process.argv.slice(2);

if (!args[0] || args[0] === '--help') {
    console.log('Usage: node download_dukascopy.js <instrument> [h1] [730] [--save-as NAME]');
    process.exit(0);
}

let [instrument, timeframe = 'h1', days = 730] = args;
let saveAs = null;

const saveAsIndex = args.indexOf('--save-as');
if (saveAsIndex > -1) saveAs = args[saveAsIndex + 1];

if (TIMEFRAMES[args[2]]) timeframe = args[2];
if (!isNaN(args[2])) days = parseInt(args[2]);
if (!isNaN(args[1]) && !TIMEFRAMES[args[1]]) { days = parseInt(args[1]); timeframe = 'h1'; }

downloadData(instrument, timeframe, days, saveAs).catch(err => {
    console.error(`Erreur: ${err.message}`);
    process.exit(1);
});