const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 10000;

app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Ensure uploads directory exists
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
  console.log("Created 'uploads' folder");
}

// Multer setup
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename: (req, file, cb) => cb(null, file.originalname)
});

const upload = multer({ storage });

const multiUpload = upload.fields([
  { name: 'mixtureImage', maxCount: 1 },
  { name: 'ingredientImages', maxCount: 10 }
]);

// Main analysis route
app.post('/run-analysis', multiUpload, (req, res) => {
  const concentrationString = req.body.concentrations || "";
  const mixtureImagePath = path.join(__dirname, req.files['mixtureImage'][0].path);
  const ingredientImagePaths = req.files['ingredientImages'].map(file => path.join(__dirname, file.path));

  console.log("Mixture Image:", req.files.mixtureImage[0].path);
  console.log("Ingredient Images:", req.files.ingredientImages.map(f => f.path));

  const pythonArgs = [
    path.join(__dirname, 'TLC-Final-Python', 'main.py'),
    mixtureImagePath,
    ...ingredientImagePaths,
    '--concentrations',
    concentrationString
  ];

  const pythonProcess = spawn('python', pythonArgs);

  let resultData = '';
  let errorData = '';

  pythonProcess.stdout.on('data', (data) => {
    resultData += data.toString();
    console.log(`stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    errorData += data.toString();
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    // Clean up uploaded files
    fs.unlinkSync(mixtureImagePath);
    ingredientImagePaths.forEach(p => fs.unlinkSync(p));

    try {
      const resultJson = JSON.parse(resultData);

      if (!resultJson.Solved) {
        return res.status(400).send({ error: 'No valid solution found', result: resultJson });
      }

      res.send(resultJson);
    } catch (err) {
      console.error('Failed to parse JSON from Python:', err);
      res.status(500).send({
        error: 'Failed to parse result from analysis',
        details: errorData || err.message
      });
    }

    console.log('Python script finished with code', code);
  });
});

// Serve HTML frontend
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
//app.listen(PORT, () => {
//  console.log(`Server is running at http://localhost:${PORT}`);
//});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});