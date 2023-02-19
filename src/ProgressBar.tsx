interface ProgressBarProps {
  completed: number;
  bgColor?: string;
  title?: string;
}
const ProgressBar = ({
  completed,
  bgColor = "#1e88e5",
  title = "Progress",
}: ProgressBarProps) => {
  const containerStyles = {
    width: "100%",
    backgroundColor: "#e0e0de",
    borderRadius: 50,
  } as React.CSSProperties;

  const fillerStyles = {
    height: "100%",
    width: `${completed}%`,
    backgroundColor: `${bgColor}`,
    borderRadius: "inherit",
    textAlign: "right",
  } as React.CSSProperties;

  const labelStyles = {
    padding: 5,
    color: "white",
    fontWeight: "bold",
  } as React.CSSProperties;

  return (
    <div style={containerStyles}>
      <div style={fillerStyles}>
        <span style={labelStyles}>{`${title}:${completed}%`}</span>
      </div>
    </div>
  );
};

export default ProgressBar;
