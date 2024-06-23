import { ReactNode } from "react";

export function FormRow({ label, error, children }: { label: string; error: string; children: ReactNode }) {
  return (
    <div>
      {label && (
        <label
          className="lg:ml-2 lg:mr-2 ml-1 mr-1"
          htmlFor={children && (children as React.ReactElement).props?.id}>
          {label}
        </label>
      )}
      {children}
      {error && <span className="ml-2 text-red-600">{error}</span>}
    </div>
  );
}
export default FormRow;
