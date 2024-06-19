import { ReactNode } from "react";

export function FormRow({ label, error, children }: { label: string; error: string; children: ReactNode }) {
  return (
    <div>
      {label && (
        <label
          className="ml-2 mr-2"
          htmlFor={children && (children as React.ReactElement).props?.id}>
          {label}
        </label>
      )}
      {children}
      {error && <span>{error}</span>}
    </div>
  );
}
export default FormRow;
