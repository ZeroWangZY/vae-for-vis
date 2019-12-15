// 文件超过1g时设大点内存：node scagnostics/labeller.js --max-old-space-size=8192

const fs = require("fs");
const path = require("path");
const eol = require('eol');
const scagnosticIndices = require("./indices");

const IMAGE_DIM = 112;

const useReal = false;
const filePrefix = useReal ? "real_" : "";

let limit = 10;

const pathForReadOfScatters = path.join(__dirname, `../data/${filePrefix}scatters_points_${IMAGE_DIM}.json`);
const pathForWriteOfScatters = path.join(__dirname, `../data/${filePrefix}scatters_labels_${IMAGE_DIM}.json`);
const pathForWriteOfValidLabels = path.join(__dirname, `../data/${filePrefix}scatters_valid_labels_${IMAGE_DIM}.json`);

function readBigJSON(filename, cb) {
  let output = "";
  const readStream = fs.createReadStream(filename);

  readStream.on("data", function(chunk) {
    output += eol.auto(chunk.toString('utf8'));
  });

  readStream.on('end', function() {
    console.log('finished reading');
    // TODO
    cb(JSON.parse(output));
  });
}

function isIndicesValid(indices) {
  const keys = Object.keys(indices);
  let isValid = false;

  for (let i = 0, len = keys.length; i < len; i += 1) {
    if (indices[keys[i]] >= 0.9) {
      isValid = true;
      break;
    }
  }

  return isValid;
}

function labelScatters(scatters) {
  let labels = [];
  let validIndices = [];

  const prefix = "getC";
  const prefixTester = new RegExp(`^${prefix}`);
  const funcNames = Object.keys(scagnosticIndices).filter(funcName => prefixTester.test(funcName));

  for (let i = 0, len = scatters.length; i < len; i += 1) {
    console.log(`labelling scatter #${i} / ${len}`);

    let labelInfo = {};
    scatter = scatters[i];

    deduplicatedScatter = scagnosticIndices.deduplicateDataset(scatter);

    normalizedScatter = scagnosticIndices.normalizeDataset(deduplicatedScatter);

    let indices = {};
    let label = -1;
    let maxIndex = 0;
    funcNames.forEach((funcName, j) => {
      const indexName = funcName.slice(prefix.length);
      // 去重后只有1个点的直接全给-1了，不算了
      const index = normalizedScatter.length < 2 ? -1 : scagnosticIndices[funcName](normalizedScatter);

      if (index > maxIndex) {
        maxIndex = index;
        label = j;
      }
      indices[indexName] = index;
    });

    labelInfo["indices"] = indices;
    labelInfo["label"] = label;

    labels.push(labelInfo);

    if (isIndicesValid(indices)) {
      validIndices.push(i);
      const len1 = validIndices.length;
      console.log(`valid scatters: ${len1} / ${limit}`);
      // if (len1 > limit) {
      //   break;
      // }
    }
  }

  fs.writeFileSync(pathForWriteOfScatters, JSON.stringify(labels));
  fs.writeFileSync(pathForWriteOfValidLabels, JSON.stringify(validIndices));
}

readBigJSON(pathForReadOfScatters, labelScatters);
