import { useState } from "react";
import { pipeline, env } from "@huggingface/transformers";
import { Card, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Upload, Loader2, Recycle, Leaf, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

type WasteCategory = "recyclable" | "organic" | "hazardous" | "unknown";

interface ClassificationResult {
  category: WasteCategory;
  confidence: number;
  description: string;
  tips: string[];
  scores: { recyclable: number; organic: number; hazardous: number };
}

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [rawPredictions, setRawPredictions] = useState<any[] | null>(null);
  const [history, setHistory] = useState<ClassificationResult[]>([]);

  // keyword lists
  const recyclableKeywords = [
    "bottle","plastic","container","jar","can","aluminum","tin","glass","box","carton","paper","cardboard",
    "newspaper","magazine","envelope","wrapper","package","bag","cup","plate"
  ];
  const organicKeywords = [
    "apple","banana","orange","fruit","vegetable","food","bread","meat","fish","egg","dairy","cheese","yogurt",
    "peel","core","leaf","plant","salad","soup","coffee","tea","tomato","onion"
  ];
  const hazardousKeywords = [
    "battery","cellphone","phone","mobile","laptop","computer","tv","monitor","bulb","light bulb",
    "led","fluorescent","paint","chemical","cleaner","detergent","medicine","pill","syringe","thermometer",
    "motor oil","gasoline","pesticide","insecticide","spray","aerosol","printer","toner","cartridge"
  ];

  const labelTokens = (label: string) =>
    label.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(Boolean);

  const classifyWaste = (predictions: any[]): ClassificationResult => {
    let scores = { recyclable: 0, organic: 0, hazardous: 0 };

    predictions.forEach((pred) => {
      const label = String(pred.label || "").toLowerCase();
      const score = typeof pred.score === "number" ? pred.score : 0;
      const tokens = labelTokens(label);

      tokens.forEach((t) => {
        if (recyclableKeywords.includes(t)) scores.recyclable += score * 1.6;
        if (organicKeywords.includes(t)) scores.organic += score * 1.6;
        if (hazardousKeywords.includes(t)) scores.hazardous += score * 1.6;
      });

      recyclableKeywords.forEach((k) => { if (label.includes(k)) scores.recyclable += score * 0.9; });
      organicKeywords.forEach((k) => { if (label.includes(k)) scores.organic += score * 0.9; });
      hazardousKeywords.forEach((k) => { if (label.includes(k)) scores.hazardous += score * 0.9; });

      if (label.includes("food") || label.includes("edible")) scores.organic += score * 0.8;
      if (label.includes("electronic") || label.includes("device") || label.includes("battery")) scores.hazardous += score * 1.2;
      if (label.includes("container") || label.includes("packaging") || label.includes("wrapper")) scores.recyclable += score * 0.6;
    });

    const total = Math.max(1e-6, scores.recyclable + scores.organic + scores.hazardous);
    const norm = {
      recyclable: scores.recyclable / total,
      organic: scores.organic / total,
      hazardous: scores.hazardous / total,
    };

    const order = Object.entries(norm).sort((a, b) => b[1] - a[1]);
    const topName = order[0][0] as keyof typeof norm;
    const topScore = order[0][1];
    const secondScore = order[1][1];

    if (topScore < 0.18 || topScore / (secondScore + 1e-9) < 1.25) {
      return {
        category: "unknown",
        confidence: Math.round(topScore * 100),
        description: "The model is uncertain. Try a clearer photo or different angle.",
        tips: [
          "Use a close-up photo with good lighting",
          "Remove background clutter",
          "Try another photo showing the object clearly",
        ],
        scores: norm,
      };
    }

    const category =
      topName === "organic" ? "organic" :
      topName === "hazardous" ? "hazardous" : "recyclable";

    const confidence = Math.round(Math.min(topScore * 100 + (topScore - secondScore) * 50, 99));

    const descriptions = {
      recyclable: "This item appears recyclable. Clean it and place in recycling.",
      organic: "This looks like organic waste — compost if available.",
      hazardous: "This item may be hazardous — follow proper disposal guidelines.",
    };

    const tips = {
      recyclable: [
        "Remove food residue",
        "Check local recycling rules",
        "Separate materials if required",
      ],
      organic: [
        "Good for composting",
        "Remove non-organic packaging",
        "Use municipal organics collection",
      ],
      hazardous: [
        "Do not put in household trash",
        "Find local hazardous-waste collection",
        "Some retailers accept electronics/batteries",
      ],
    };

    return {
      category,
      confidence,
      description: descriptions[category],
      tips: tips[category],
      scores: norm,
    };
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedImage(file);
    setResult(null);
    setRawPredictions(null);

    const reader = new FileReader();
    reader.onload = (ev) => setImagePreview(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const analyzeImage = async () => {
    if (!selectedImage) {
      toast.error("Select an image first");
      return;
    }
    setIsProcessing(true);
    setResult(null);
    setRawPredictions(null);
    try {
      toast.info("Analyzing...");
      const device = (navigator as any)?.gpu ? "webgpu" : "wasm";
      const classifier = await pipeline(
        "image-classification",
        "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
        { device }
      );

      const imageUrl = URL.createObjectURL(selectedImage);
      const preds = await classifier(imageUrl);
      URL.revokeObjectURL(imageUrl);

      console.log("Raw predictions:", preds);
      setRawPredictions(preds);

      const res = classifyWaste(preds);
      setResult(res);

      setHistory((prev) => {
        const newHist = [res, ...prev];
        return newHist.slice(0, 5);
      });

      toast.success("Done");
    } catch (err) {
      console.error("Analyze error:", err);
      toast.error("Analysis failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const getCategoryIcon = (category: WasteCategory) =>
    category === "organic" ? (
      <Leaf className="w-8 h-8 text-green-600" />
    ) : category === "hazardous" ? (
      <AlertTriangle className="w-8 h-8 text-red-600" />
    ) : category === "recyclable" ? (
      <Recycle className="w-8 h-8 text-blue-600" />
    ) : (
      <div className="w-8 h-8 text-gray-600">?</div>
    );

  // Dynamic eco tips
  const ecoTips: Record<WasteCategory, string[]> = {
    recyclable: [
      "Clean and separate recyclables properly",
      "Use reusable containers instead of disposables",
      "Avoid mixing recyclables with organic/hazardous waste",
    ],
    organic: [
      "Compost food and plant waste",
      "Avoid throwing food in trash",
      "Use kitchen scraps for home gardening",
    ],
    hazardous: [
      "Do not throw batteries, electronics, or chemicals in trash",
      "Take hazardous items to certified disposal centers",
      "Educate family and friends about safe disposal",
    ],
    unknown: [
      "Try capturing a clearer image for proper classification",
      "Check local disposal guidelines",
      "Separate waste until certain of category",
    ],
  };

  return (
    <div className="min-h-screen w-screen bg-gradient-to-br from-green-50 via-blue-50 to-pink-50 px-6 md:px-12 py-6 flex flex-col">
      {/* Header */}
      <header className="text-center mb-6 flex-none">
        <h1 className="text-3xl md:text-4xl font-bold animate-pulse">Waste Segregation AI</h1>
        <p className="text-gray-600 text-sm md:text-lg mt-1">Instant classification, tips & confidence</p>
      </header>

      {/* Main content */}
      <div className="flex flex-1 gap-6">
        {/* Left: Upload + Eco Tips */}
        <div className="flex-1 flex flex-col gap-4">
          {/* Upload Card */}
          <Card className="p-6 w-full hover:scale-105 transition-transform shadow-lg flex flex-col items-center justify-center">
            <div className="mx-auto w-16 h-16 bg-green-500 rounded-full flex items-center justify-center animate-bounce">
              <Upload className="w-8 h-8 text-white" />
            </div>
            <h3 className="mt-3 text-lg font-semibold">Upload Image</h3>
            <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" id="imgfile" />
            <label htmlFor="imgfile" className="inline-flex items-center gap-2 px-4 py-2 mt-3 bg-blue-600 text-white rounded cursor-pointer hover:bg-blue-700">
              <Upload className="w-4 h-4" /> Choose Image
            </label>

            {imagePreview && (
              <>
                <img src={imagePreview} alt="preview" className="mt-3 rounded max-h-64 shadow-lg" />
                <Button onClick={analyzeImage} disabled={isProcessing} className="mt-3 w-full bg-green-600 hover:bg-green-700">
                  {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : "Analyze"}
                </Button>
              </>
            )}
          </Card>

          {/* Eco Tips */}
          <Card className="p-4 hover:scale-105 transition-transform shadow-lg flex-1">
            <h4 className="font-semibold text-sm">Eco-Friendly Habits</h4>
            <ul className="list-disc ml-5 text-xs mt-2 space-y-1">
              {(result ? ecoTips[result.category] : [
                "Carry reusable bags, bottles, and cutlery",
                "Compost organic waste whenever possible",
                "Donate old clothes and electronics instead of throwing them",
                "Segregate waste at source into wet/dry/hazardous",
              ]).map((tip, i) => (
                <li key={i}>{tip}</li>
              ))}
            </ul>
          </Card>
        </div>

        {/* Right: Results */}
        <div className="flex-1 flex flex-col gap-4">
          {/* Classification Result */}
          {result && (
            <Card className="p-6 hover:scale-105 transition-transform shadow-lg flex flex-col justify-center flex-1">
              <div className="flex gap-3 items-center">
                <div className="p-3 bg-gray-100 rounded-full">{getCategoryIcon(result.category)}</div>
                <div>
                  <h3 className="text-lg font-semibold capitalize">{result.category}</h3>
                  <p className="text-sm text-gray-600">{result.confidence}% confidence</p>
                  <p className="text-xs md:text-sm mt-1">{result.description}</p>
                </div>
              </div>
              <ul className="mt-2 list-disc pl-5 text-xs md:text-sm">
                {result.tips.map((t, i) => <li key={i}>{t}</li>)}
              </ul>
            </Card>
          )}

          {/* Confidence Chart */}
          {result && (
            <Card className="p-4 hover:scale-105 transition-transform shadow-lg flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={[
                    { name: "Recyclable", value: Math.round(result.scores.recyclable * 100) },
                    { name: "Organic", value: Math.round(result.scores.organic * 100) },
                    { name: "Hazardous", value: Math.round(result.scores.hazardous * 100) },
                  ]}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* History + Raw Predictions */}
          <div className="flex gap-4 flex-wrap">
            {rawPredictions && rawPredictions.length > 0 && (
              <Card className="p-4 hover:scale-105 transition-transform shadow-lg flex-shrink max-w-[48%]">
                <h4 className="text-sm font-semibold mb-2">Raw Predictions</h4>
                <ol className="list-decimal ml-5 text-xs max-h-32 overflow-y-auto">
                  {rawPredictions.slice(0, 5).map((p: any, i: number) => (
                    <li key={i}>
                      <strong>{p.label}</strong> — {(p.score * 100).toFixed(1)}%
                    </li>
                  ))}
                </ol>
              </Card>
            )}

            {history.length > 0 && result && (
              <Card className="p-4 hover:scale-105 transition-transform shadow-lg flex-shrink max-w-[48%]">
                <h4 className="text-sm font-semibold mb-2">Recently Classified</h4>
                <ul className="list-disc ml-5 text-xs max-h-32 overflow-y-auto">
                  {history.slice(0, 5).map((h, i) => (
                    <li key={i}>
                      <strong className="capitalize">{h.category}</strong> — {h.confidence}% ({h.description})
                    </li>
                  ))}
                </ul>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
