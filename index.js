const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
//const PORT = 3000;
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
      cb(null, file.originalname);  // <-- keep original file name
    }
  });
  
  const upload = multer({ storage: storage });

const multiUpload = upload.fields([
  { name: 'mixtureImage', maxCount: 1 },
  { name: 'ingredientImages', maxCount: 10 } // adjust if you expect more than 10
]);

app.post('/run-analysis', multiUpload, (req, res) => {
    const concentrationString = req.body.concentrations || "";
    const mixtureImagePath = path.join(__dirname, req.files['mixtureImage'][0].path);
    const ingredientImagePaths = req.files['ingredientImages'].map(file => path.join(__dirname, file.path));
  
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
    });
  
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error(`stderr: ${data}`);
    });
  
    pythonProcess.on('close', (code) => {
      // Clean up uploaded files
      fs.unlinkSync(mixtureImagePath);
      ingredientImagePaths.forEach(p => fs.unlinkSync(p));
  
      if (code !== 0 || !resultData) {
        res.status(500).send({ error: 'Analysis failed', details: errorData || 'Unknown error' });
      } else {
        res.send({ csv: resultData });
      }
    });
  });
  
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'plotly_csv_2.html'));
});

//app.listen(PORT, () => {
//  console.log(`Server is running at http://localhost:${PORT}`);
//});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running on port ${PORT}`);
});