"use client";

import { Toaster, toast } from "sonner";
import Form from "./Form";
import { useState } from "react";
import Predict from "./Predict";

export default function Home() {
  const [data, setData] = useState<string>("");

  const renderResult = () => {
    if (data === "0") {
      return (
        <Predict
          text="Unfortunately, you would not survive the Titanic sink."
          src="https://media1.tenor.com/m/qYBbEjA6_cIAAAAd/a-night-to-remember-movie-a-night-to-remember.gif"
          alt="Gif of a ship sunking"
        />
      );
    } else if (data === "1") {
      return (
        <Predict
          text="Congratulations! You would survive the Titanic sink."
          src="https://media1.tenor.com/m/0yP-5qSBo_sAAAAC/ben-hur-ben-hur-movie.gif"
          alt="Gif of a survivor at the beach"
        />
      );
    } else {
      return toast.error("An unknown error occurred", { duration: 5000 });
    }
  };
  return (
    <>
      <Toaster richColors />
      <main>
        <h1 className="md:mt-12 md:mb-12 mt-6 mb-6 text-center text-4xl md:text-6xl text-balance">
          Would you survive the Titanic sink?
        </h1>
        <Form setData={setData}></Form>
        {data && renderResult()}
      </main>
    </>
  );
}
