export default function Predict({ text, src, alt }: { text: string; src: string; alt: string }) {
  return (
    <div className="mt-12 flex items-center justify-center text-center flex-col">
      <h2 className="mb-12">{text}</h2>
      <img
        src={src}
        alt={alt}
      />
    </div>
  );
}
