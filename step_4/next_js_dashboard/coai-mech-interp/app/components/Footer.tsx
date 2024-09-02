export default function Footer() {
  return (
    <footer className="bg-gray-800 p-4 text-center text-white"> {/* Set text color to white */}
      <p>&copy; {new Date().getFullYear()} COAI Research. All rights reserved.</p>
    </footer>
  );
}