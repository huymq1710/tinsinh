import axios from "axios";
import "./App.css";
import React, { Component } from "react";
import ReactDOM from "react-dom";
import Modal from "react-modal";
axios.defaults.baseURL = "http://localhost:5000";
class App extends Component {
  state = {
    // Initially, no file is selected
    selectedFile: null,
    text: "",
    type: "",
    key: "",
    modalshow: false,
    textRes: "",
    arrtext: [],
  };

  // On file select (from the pop up)
  onFileChange = (event) => {
    // Update the state
    this.setState({ selectedFile: event.target.files[0] });
  };

  // On file upload (click the upload button)
  onFileUpload = async () => {

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
    // console.log(this.state.selectedFile);
    // console.log(formData);
    // Request made to the backend api
    // Send formData object
    let res = await axios.post("/", formData);

    let newarr = res.data.data.split(">");
    let newdata = [];
    newarr.map((e) => {
      if (e.includes("Carbonylation Site, as follows:")) {
        let datatest = e.split("Carbonylation Site, as follows:");
        if (datatest[0] == "") {
          datatest[0] = "Carbonylation Site, as follows:";
        } else if (datatest[1] == "") {
          datatest[1] = "Carbonylation Site, as follows:";
        }
        newdata.push(datatest[0], datatest[1]);
      } else if (e.includes("Non-carbonylation Site, as follows:")) {
        console.log(e);
        let datatest = e.split("Non-carbonylation Site, as follows:");
        console.log(datatest[1] == "");
        if (datatest[0] != "") {
          datatest[1] = "Non-carbonylation Site, as follows:";
        } else if (datatest[1] != "") {
          datatest[0] = "Non-carbonylation Site, as follows:";
        }
        console.log(datatest);
        newdata.push(datatest[0], datatest[1]);
      } else {
        newdata.push(e);
      }
    });
    console.log(newdata);
    this.setState({
      textRes: res.data.data,
      modalshow: true,
      arrtext: newdata,
    });
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
        <div className="header">
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
            Đề tài: Dự đoán Protein Carbonylation
          </p>
        </div>
        <div className="middle">
          <textarea
            name="Text1"
            cols="100"
            rows="25"
            style={{ margin: 10 }}
            onChange={(e) => {
              this.setState({ text: e.target.value });
            }}
            placeholder="Enter sequence(s) in FASTA format:"
          ></textarea>
        </div>
        <table className="custom-table">
          <tr>
            <td>Or upload a file in FASTA format</td>
            <td><input type="file" onChange={this.onFileChange} /></td>
          </tr>
          <tr>
            <td>Please chooose a suitable type for prediction</td>
            <td><select onChange={(e) => {
              this.setState({
                key: e.target.value
              })
            }} >
              <option value="K">Carbonylation Sites: K (lysine) </option>
              <option value="P">Carbonylation Sites: P (proline) </option>
              <option value="T">Carbonylation Sites: T (threonine) </option>
              <option value="R">Carbonylation Sites: R (arginine) </option>
            </select></td>
          </tr>
        </table>

        <div className="footer">
          <button className="button" onClick={this.onFileUpload}>
            Submit
          </button>
        </div>
        <Modal isOpen={this.state.modalshow} contentLabel="Example Modal">
          <div style={{ display: "flex", flexDirection: "row" }}>
            <h2 style={{ flex: 1 }}>Kết quả dự đoán </h2>{" "}
            <button
              onClick={() => {
                this.setState({ modalshow: false });
              }}
            >
              Đóng
            </button>
          </div>
          <div>
            {this.state.arrtext.map((e) => {
              if (e == "Carbonylation Site, as follows:") {
                return <b>{e}</b>;
              } else if (e == "Non-carbonylation Site, as follows:") {
                return <b>{e}</b>;
              } else {
                return <p>{e}</p>;
              }
            })}
          </div>
        </Modal>
      </div>
    );
  }
}

export default App;
