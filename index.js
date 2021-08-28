const express = require("express");
const bodyParser = require("body-parser");
const multer = require("multer");
const app = express();
const fs = require("fs");
var upload = multer({ dest: "input/" });
const util = require("util");
var cors = require("cors");
const exec = util.promisify(require("child_process").exec);
//CREATE EXPRESS APP
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());
//ROUTES WILL GO HERE
app.post("/", upload.single("myFile"), async function (req, res) {
  let a = req.body;
  console.log(req.file);
  if (a.string) {
    let filename = new Date().getTime().toString();
    fs.writeFileSync(`./input/${filename}.fsa`, a.string);

    const { stdout, stderr } = await exec(
      `python3 iCarPS_offline.py ${a.key} ./input/${filename}.fsa`
    );

    let result = fs.readFileSync(`./output/${filename}_finalresult.txt`);
    //console.log(result.toString("utf8"));
    res.send({ data: result.toString("utf8") });

    // res.sendFile(
    //   "/home/thangdp/tinsinh/iCarPS/test/3940d62ddf5b514cd990b1ee5c8f4fcc"
    // );
  } else {
    console.log(req.body.key);
    const { stdout, stderr } = await exec(
      `python3 iCarPS_offline.py ${a.key} ./input/${req.file.filename}`
    );
    let result = fs.readFileSync(
      `./output/${req.file.filename}_finalresult.txt`
    );
    //console.log(result.toString("utf8"));
    res.send({ data: result.toString("utf8") });
  }

  // fs.readFile(req.file, "utf8", (err, data) => {
  //   if (err) {
  //     console.error(err);
  //     return;
  //   }
  //   console.log(data);
  // });
});

app.listen(5000, () => console.log("Server started on port 3000"));
