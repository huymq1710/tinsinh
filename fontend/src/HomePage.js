import React from "react";

class HomePage extends Component {
  state = {
    // Initially, no file is selected
    selectedFile: null,
    text: "",
    type: "",
  };

  // On file select (from the pop up)
  onFileChange = (event) => {
    // Update the state
    this.setState({ selectedFile: event.target.files[0] });
  };

  // On file upload (click the upload button)
  onFileUpload = () => {
    // Create an object of formData
    const formData = new FormData();

    // Update the formData object
    formData.append(
      "myFile",
      this.state.selectedFile
      // this.state.selectedFile.name
    );
    formData.append(
      "key",
      this.state.key
      // this.state.selectedFile.name
    );
    formData.append(
      "string",
      this.state.text
      // this.state.selectedFile.name
    );
    // Details of the uploaded file
    console.log(this.state.selectedFile);
    console.log(formData);
    // Request made to the backend api
    // Send formData object
    axios.post("/", formData);
  };

  // File content to be displayed after
  // file upload is complete
  fileData = () => {
    if (this.state.selectedFile) {
      return (
        <div>
          <h2>File Details:</h2>

          <p>File Name: {this.state.selectedFile.name}</p>

          <p>File Type: {this.state.selectedFile.type}</p>

          <p>
            Last Modified:{" "}
            {this.state.selectedFile.lastModifiedDate.toDateString()}
          </p>
        </div>
      );
    } else {
      return (
        <div>
          <br />
          <h4>Choose before Pressing the Upload button</h4>
        </div>
      );
    }
  };

  render() {
    return (
      <div className="Khung">
        <div>
          <p
            style={{
              textAlign: "center",
              fontSize: 30,
              fontWeight: "bold",
              color: "antiquewhite",
            }}
          >
            Bài tập cuối kỳ Tin Sinh
          </p>
          <p
            style={{
              textAlign: "center",
              fontSize: 30,
              fontWeight: "bold",
              color: "antiquewhite",
            }}
          >
            Nhóm TungBT và những người bạn
          </p>
        </div>
        <div style={{ display: "flex" }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            Enter sequence(s) in FASTA format:
            <textarea
              name="Text1"
              cols="100"
              rows="25"
              style={{ margin: 10 }}
              onChange={(e) => {
                console.log(e.target.value);
                this.setState({ text: e.target.value });
              }}
            ></textarea>
            Or upload a file in FASTA format :
            <input type="file" onChange={this.onFileChange} />
          </div>
          <div style={{ display: "flex" }}>
            <div style={{ margin: 30 }}>
              Please choose a suitable type for prediction
              <div>
                <input
                  type="radio"
                  id="html"
                  name="fav_language"
                  value="HTML"
                />
                <label for="html">Carbonylation Sites: K (lysine)</label>
                <br />
                <input type="radio" id="css" name="fav_language" value="CSS" />
                <label for="css">Carbonylation Sites: P (proline)</label>
                <br />
                <input type="radio" id="css" name="fav_language" value="CSS" />
                <label for="css">Carbonylation Sites: T (threonine)</label>
                <br />
                <input
                  type="radio"
                  id="javascript"
                  name="fav_language"
                  value="JavaScript"
                />
                <label for="javascript">
                  Carbonylation Sites: R (arginine)
                </label>
              </div>
              <button style={{ margin: 30 }} onClick={this.onFileUpload}>
                Submit
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default HomePage;
