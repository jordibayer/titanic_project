"use client";

import { useState } from "react";
import FormRadius from "./FormRadius";

export default function Form() {
  const [rangeValue, setRangeValue] = useState(25);
  const handleRangeChange = (e: any) => {
    setRangeValue(e.target.value);
  };

  const [familyValue, setFamilyValue] = useState(2);
  const handleFamilyValue = (e: any) => {
    setFamilyValue(e.target.value);
  };
  return (
    <form className="flex flex-col items-center space-y-4 border-[1px] rounded-md border-gray-700 bg-zinc-900 p-12 lg:mx-60 mx-5">
      <div className="flex">
        <div className="flex items-center border-[1px] rounded-md p-1 border-gray-700">
          <label className="mr-2">Sex:</label>
          <FormRadius
            name="sex"
            id="Male"
          />
          <FormRadius
            name="sex"
            id="Female"
          />
        </div>
        <div className="flex items-center ml-5 border-[1px] rounded-md p-1 border-gray-700">
          <label className="mr-2">Age:</label>
          <input
            className="mr-2"
            type="range"
            min="0"
            max="100"
            defaultValue="25"
            onChange={handleRangeChange}
          />
          <output>{rangeValue}</output>
        </div>
      </div>
      <div className="flex items-center border-[1px] rounded-md p-1 border-gray-700">
        <label className="mr-2">How many family members are travelling with you:</label>
        <input
          className="mr-2"
          type="range"
          min="0"
          max="15"
          defaultValue="2"
          onChange={handleFamilyValue}
        />
        <output>{familyValue}</output>
      </div>
      <div className="flex items-center ml-5 border-[1px] rounded-md p-1 border-gray-700">
        <label className="mr-2">Type of ticket:</label>
        <FormRadius
          name="pclass"
          id="Cheapest"
        />
        <FormRadius
          name="pclass"
          id="Middle Class"
        />
        <FormRadius
          name="pclass"
          id="Luxury"
        />
      </div>
      <input
        type="submit"
        className="border-[1px] rounded-md px-4 py-2 bg-gray-800 hover:bg-gray-600 border-gray-700"
      />
    </form>
  );
}
