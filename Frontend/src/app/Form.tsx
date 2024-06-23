"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import FormSection from "./FormSection";
import { toast } from "sonner";

const URL = process.env.NEXT_PUBLIC_API_URL
  ? `https://${process.env.NEXT_PUBLIC_API_URL}/api`
  : "http://localhost:8000/api";

interface FormData {
  sex: string;
  age: number;
  familySize: number;
  pclass: string;
}

interface TransformedData {
  sex: number;
  age: number;
  familySize: number;
  pclass: number;
}

interface FormProps {
  setData: React.Dispatch<React.SetStateAction<any>>;
}

export default function Form({ setData }: FormProps) {
  const [isUpdating, setIsUpdating] = useState(false);
  const [rangeValue, setRangeValue] = useState(25);
  const [familyValue, setFamilyValue] = useState(0);

  const { register, handleSubmit, formState } = useForm<FormData>();
  const { errors } = formState;

  const transformData = (data: FormData): TransformedData => {
    return {
      sex: data.sex === "male" ? 1 : 0,
      age: data.age,
      familySize: data.familySize,
      pclass: data.pclass === "cheapest" ? 3 : data.pclass === "middleClass" ? 2 : 1,
    };
  };

  const onSubmit = async (formData: any) => {
    setIsUpdating(true);
    const transformedData = transformData(formData);
    try {
      const result = await fetch(`${URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(transformedData),
      });
      if (!result.ok) throw new Error("Error sending the info, please try again.");
      const data = await result.json();
      setData(data.data);
    } catch (error) {
      if (error instanceof Error) {
        toast.error(error.message, { duration: 5000 });
      } else {
        toast.error("An unknown error occurred", { duration: 5000 });
      }
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="flex flex-col items-center space-y-4 border-[1px] rounded-md border-gray-700 bg-zinc-900 p-12 lg:mx-60 mx-5">
      <div className="flex">
        <div className="flex items-center border-[1px] rounded-md p-1 border-gray-700">
          <label className="mr-2">Sex:</label>
          <FormSection
            label="Male"
            error={errors?.sex?.message || ""}>
            <input
              value="male"
              type="radio"
              disabled={isUpdating}
              {...register("sex", {
                required: "This field is required",
              })}
            />
          </FormSection>
          <FormSection
            label="Female"
            error={errors?.sex?.message || ""}>
            <input
              value="female"
              type="radio"
              disabled={isUpdating}
              {...register("sex", {
                required: "This field is required",
              })}
            />
          </FormSection>
        </div>
        <div className="flex items-center ml-5 border-[1px] rounded-md p-1 border-gray-700">
          <FormSection
            label="Age:"
            error={errors?.age?.message || ""}>
            <input
              className="mr-2"
              type="range"
              min="0"
              max="100"
              defaultValue="25"
              disabled={isUpdating}
              {...register("age", {
                required: "This field is required",
                min: {
                  value: 1,
                  message: "Age should be atleast 1",
                },
                onChange(event) {
                  setRangeValue(event.target.value);
                },
              })}
            />
          </FormSection>

          <output>{rangeValue}</output>
        </div>
      </div>
      <div className="flex items-center border-[1px] rounded-md p-1 border-gray-700">
        <FormSection
          label="How many family members are travelling with you:"
          error={errors?.familySize?.message || ""}>
          <input
            className="mr-2"
            type="range"
            min="0"
            max="15"
            defaultValue="0"
            disabled={isUpdating}
            {...register("familySize", {
              required: "This field is required",
              min: {
                value: 0,
                message: "Family size should be equal or greater than 0",
              },
              onChange(event) {
                setFamilyValue(event.target.value);
              },
            })}
          />
        </FormSection>

        <output>{familyValue}</output>
      </div>
      <div className="flex items-center ml-5 border-[1px] rounded-md p-1 border-gray-700">
        <label className="mr-2">Type of ticket:</label>
        <FormSection
          label="Cheapest"
          error={errors?.pclass?.message || ""}>
          <input
            value="cheapest"
            type="radio"
            disabled={isUpdating}
            {...register("pclass", {
              required: "This field is required",
            })}
          />
        </FormSection>
        <FormSection
          label="Middle Class"
          error={errors?.pclass?.message || ""}>
          <input
            value="middleClass"
            type="radio"
            disabled={isUpdating}
            {...register("pclass", {
              required: "This field is required",
            })}
          />
        </FormSection>
        <FormSection
          label="Luxury"
          error={errors?.pclass?.message || ""}>
          <input
            value="luxury"
            type="radio"
            disabled={isUpdating}
            {...register("pclass", {
              required: "This field is required",
            })}
          />
        </FormSection>
      </div>
      <input
        type="submit"
        disabled={isUpdating}
        className="border-[1px] rounded-md px-4 py-2 bg-gray-800 hover:bg-gray-600 border-gray-700"
      />
    </form>
  );
}
