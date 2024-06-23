export default function Predict({ text, src, alt }: { text: string; src: string; alt: string }) {
  return (
    <div className="lg:mt-12 mt-6 flex items-center justify-center flex-col">
      <h2 className="mb-6 text-center text-balance text-3xl md:text-5xl px-2">{text}</h2>
      <img
        className="px-5 lg:px-0"
        src={src}
        alt={alt}
      />
    </div>
  );
}
