const util = require("util");
const exec = util.promisify(require("child_process").exec);

async function ls() {
  const { stdout, stderr } = await exec(
    "python iCarPS_offline.py K ./input/test.txt"
  );
  console.log("stdout:", stdout);
  console.log("stderr:", stderr);
}
ls();
