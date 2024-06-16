export default function FormRadius({ name, id }: { name: string; id: string }) {
  return (
    <>
      <input
        type="radio"
        name={name}
        id={id}
      />
      <label
        htmlFor={id}
        className="ml-2 mr-2">
        {id}
      </label>
    </>
  );
}
