const fs = require("fs");
const path = require("path");
const scagnosticIndices = require("./indices");

const IMAGE_DIM = 112;

const pathForReadOfScatters = path.join(__dirname, `../data/scatters_points_${IMAGE_DIM}.json`);
const pathForWriteOfScatters = path.join(__dirname, `../data/scatters_labels_${IMAGE_DIM}.json`);

let scatters = JSON.parse(fs.readFileSync(pathForReadOfScatters));

let labels = [];

const prefix = "getC";
const prefixTester = new RegExp(`^${prefix}`);
const funcNames = Object.keys(scagnosticIndices).filter(funcName => prefixTester.test(funcName));

for (let i = 0, len = scatters.length; i < len; i += 1) {
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
}

<<<<<<< HEAD
fs.writeFileSync(pathForWriteOfScatters, JSON.stringify(labels, null, 4))
=======
fs.writeFileSync(pathForWriteOfScatters, JSON.stringify(labels, null, 4));
>>>>>>> dev/lbr
