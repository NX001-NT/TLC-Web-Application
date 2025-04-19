const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
//const PORT = 3000;
const PORT = process.env.PORT || 10000;

app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

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
      console.log(`stdout: ${data.toString()}`);
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

      console.log('Python script finished with code', code);
      console.log('Final result:', resultData);
    });
  });
  
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

//app.listen(PORT, () => {
//  console.log(`Server is running at http://localhost:${PORT}`);
//});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});