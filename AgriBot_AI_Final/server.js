
const express = require("express");
const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const multer = require("multer");

const app = express();
const PORT = 3000;
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 }
});

let model = null;

async function loadAIModel() {
  const modelPath = path.join(__dirname, "model_v2", "model.json");
  const modelJSON = JSON.parse(fs.readFileSync(modelPath, "utf8"));

  const weightsPath = path.join(__dirname, "model_v2", "weights.bin");
  const weightsBuffer = fs.readFileSync(weightsPath);

  const modelArtifacts = {
    modelTopology: modelJSON.modelTopology,
    weightSpecs: modelJSON.weightsManifest[0].weights,
    weightData: weightsBuffer.buffer.slice(
      weightsBuffer.byteOffset,
      weightsBuffer.byteOffset + weightsBuffer.byteLength
    )
  };

  model = await tf.loadLayersModel(
    tf.io.fromMemory(modelArtifacts)
  );

  console.log("AgriBot AI Model V2 loaded successfully!");
  console.log("Model input shape:", model.inputs[0].shape);
console.log("Model output shape:", model.outputs[0].shape);
}


async function prepareImage(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .resize(224, 224, { fit: "fill" })
    .removeAlpha()
    .toColourspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });

  const imageTensor = tf.tensor3d(
    new Uint8Array(data),
    [info.height, info.width, info.channels],
    "int32"
  );

  const normalizedImage = imageTensor
    .toFloat()
    .div(127.5)
    .sub(1)
    .expandDims(0);

  imageTensor.dispose();

  return normalizedImage;
}

const metadata = JSON.parse(
  fs.readFileSync(
    path.join(__dirname, "model_v2", "metadata.json"),
    "utf8"
  )
);

async function predictPlant(imageBuffer) {
  if (!model) {
    throw new Error("AI model is not loaded yet.");
  }

  const inputTensor = await prepareImage(imageBuffer);

  try {
    const outputTensor = model.predict(inputTensor);
   const scores = await outputTensor.data();
outputTensor.dispose();

    const classNames = metadata.labels;

    const predictions = classNames.map((name, index) => ({
      className: name,
      probability: scores[index]
    }));

    predictions.sort((a, b) => b.probability - a.probability);

    const best = predictions[0];

    return {
      classification: best.className,
      confidence: Number((best.probability * 100).toFixed(1))
    };

  } finally {
    inputTensor.dispose();
  }
}

app.get("/", (req, res) => {
  res.send("AgriBot AI Service is running!");
});


app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "No image received"
      });
    }

    const result = await predictPlant(req.file.buffer);

    console.log("AgriBot Prediction:", result);

    return res.json(result);

  } catch (error) {
    console.error("Prediction error:", error);

    return res.status(500).json({
      error: "Could not analyze the image"
    });
  }
});


async function startServer() {
  try {
    await loadAIModel();

    app.listen(PORT, () => {
      console.log(`AgriBot AI Service running at http://localhost:${PORT}`);
    });

  } catch (error) {
    console.error("Failed to load AgriBot AI Model:", error);
    process.exit(1);
  }
}

startServer();
