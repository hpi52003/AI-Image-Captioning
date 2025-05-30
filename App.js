import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [originalCaption, setOriginalCaption] = useState("");
  const [translatedCaption, setTranslatedCaption] = useState("");
  const [lang, setLang] = useState("en");
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setAudioUrl(null);
    setOriginalCaption("");
    setTranslatedCaption("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Step 1: Caption + Translation
      const res = await axios.post(
        `http://localhost:8000/caption/?lang=${lang}`,
        formData
      );

      if (res.data.error) {
        setError(res.data.error);
        return;
      }

      setOriginalCaption(res.data.original_caption);
      setTranslatedCaption(res.data.translated_caption);

      // Step 2: Fetch Audio
      const audioRes = await axios.get("http://localhost:8000/audio", {
        responseType: "blob",
      });

      const url = URL.createObjectURL(audioRes.data);
      setAudioUrl(url);
    } catch (err) {
      console.error("Error:", err);
      setError("Something went wrong. Please check the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-200 to-cyan-100 flex flex-col items-center justify-center p-6">
      <div className="bg-white shadow-lg rounded-2xl p-8 w-full max-w-lg">
        <h1 className="text-3xl font-bold mb-6 text-center text-indigo-700">
          AI Image Captioning
        </h1>

        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          className="mb-4 w-full p-2 border border-gray-300 rounded"
          accept="image/*"
        />

        <select
          value={lang}
          onChange={(e) => setLang(e.target.value)}
          className="mb-4 w-full p-2 border border-gray-300 rounded"
        >
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="hi">Hindi</option>
          <option value="ja">Japanese</option>
          <option value="zh-cn">Chinese (Simplified)</option>
        </select>

        <button
          onClick={handleUpload}
          disabled={loading || !file}
          className="w-full bg-indigo-600 text-white font-semibold py-2 rounded hover:bg-indigo-700 transition disabled:opacity-50"
        >
          {loading ? "Generating..." : "Generate Caption"}
        </button>

        {error && (
          <div className="mt-4 text-red-600 font-semibold text-center">
            {error}
          </div>
        )}

        {translatedCaption && (
          <div className="mt-6 text-center">
            <h2 className="text-xl font-semibold text-gray-800">Caption</h2>
            <p className="mt-2 text-gray-600 italic">{translatedCaption}</p>
            {lang !== "en" && (
              <p className="mt-1 text-sm text-gray-500">
                (Original: {originalCaption})
              </p>
            )}

            {audioUrl && (
              <audio controls className="mt-4 mx-auto">
                <source src={audioUrl} type="audio/mpeg" />
                Your browser does not support the audio element.
              </audio>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
